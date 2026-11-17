# SPDX-License-Identifier: Apache-2.0
"""Run directly on Ascend 950 to validate the FIA mode-4 FP8 KV contract.

Exercises real dynamic quantization, strided paged scatter, prefill/decode,
and graph replay. The reference uses the same quantized KV dequantized to
BF16, so this isolates the cache/FA path from FP8 quantization error.

Load production helper definitions from source without importing vLLM's
platform plugins. All tensor operations still use real torch/torch_npu.
"""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

# Runtime SOC mapping matches vllm_ascend/device/hardware.py without importing it.
ASCEND_950_SOC_VERSION = 260


def load_source_functions(path, names, namespace):
    """Compile selected functions and their local dependencies, without imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = set(names) - functions.keys()
    if missing:
        raise RuntimeError(f"Missing production helpers in {path}: {sorted(missing)}")
    selected = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        references = {node.id for node in ast.walk(functions[name]) if isinstance(node, ast.Name)}
        pending.extend((references & functions.keys()) - selected)
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets)
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    definitions = [node for name, node in functions.items() if name in selected]
    module = ast.fix_missing_locations(ast.Module(body=[future, *constants, *definitions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def load_cache_helpers():
    """Read the synchronized source tree; this standalone test has no PD group."""
    for root in Path(__file__).resolve().parents:
        attention_dir = root / "vllm_ascend/attention"
        if (attention_dir / "attention_c8_fia.py").is_file():
            break
    else:
        raise FileNotFoundError("Run this script from the synchronized vllm-ascend source tree.")
    namespace = {
        "torch": torch,
        "torch_npu": torch_npu,
        "math": math,
        "has_kv_transfer_group": lambda: False,
    }
    load_source_functions(
        attention_dir / "attention_c8_mxfp.py",
        {"mxfp_cache_views_for_spec", "mxfp_packet_size_bytes"},
        namespace,
    )
    load_source_functions(attention_dir / "utils.py", {"notify_kv_cache_written"}, namespace)
    load_source_functions(attention_dir / "attention_c8_fia.py", {"scatter_fia_fp8_cache"}, namespace)
    return namespace


class TestFIAPerTokenNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("requires an Ascend 950 NPU")
        torch.npu.set_device(0)
        soc_version = torch_npu.npu.get_soc_version()
        print(
            f"NPU: {torch.npu.get_device_name(0)}, SOC: {soc_version}; "
            f"torch: {torch.__version__}; torch_npu: {torch_npu.__version__}"
        )
        if soc_version != ASCEND_950_SOC_VERSION:
            raise unittest.SkipTest(f"requires Ascend 950, got runtime SOC {soc_version}")
        helpers = load_cache_helpers()
        cls.scatter_fia_fp8_cache = staticmethod(helpers["scatter_fia_fp8_cache"])
        cls.mxfp_cache_views_for_spec = staticmethod(helpers["mxfp_cache_views_for_spec"])
        cls.mxfp_packet_size_bytes = staticmethod(helpers["mxfp_packet_size_bytes"])

    def make_case(self, q_len, kv_len=529, padding=0):
        torch.manual_seed(17)
        heads, kv_heads, dim, block_size = 6, 1, 128, 512
        spec = SimpleNamespace(
            num_kv_heads=kv_heads,
            head_size=dim,
            head_size_v=dim,
            block_size=block_size,
            page_size_bytes=self.mxfp_packet_size_bytes(kv_heads, block_size, dim, dim) + padding,
        )
        raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8, device="npu")
        cache = self.mxfp_cache_views_for_spec(raw, spec, block_size, fia_pseudo=True)
        key = torch.randn(kv_len, kv_heads, dim, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        magnitude = torch.exp2(torch.arange(kv_len).remainder(7).float() - 3).view(-1, 1, 1)
        key, value = (key.float() * magnitude).bfloat16().npu(), (value.float() / magnitude).bfloat16().npu()
        query = (torch.randn(q_len, heads, dim) / 4).bfloat16().npu()
        # Deliberately map consecutive logical pages to physical pages [2, 1].
        positions = torch.arange(kv_len, dtype=torch.int64)
        slots = torch.where(positions < block_size, positions + 2 * block_size, positions)
        slots = slots.npu()
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
        kwargs = {
            "query": query,
            "key": cache[0],
            "value": cache[1],
            "dequant_scale_key": cache[2],
            "dequant_scale_value": cache[3],
            "query_quant_mode": 0,
            "key_quant_mode": 4,
            "value_quant_mode": 4,
            "input_layout": "TND",
            "num_query_heads": heads,
            "num_key_value_heads": kv_heads,
            "softmax_scale": dim**-0.5,
            "block_table": torch.tensor([[2, 1]], dtype=torch.int32, device="npu"),
            "block_size": block_size,
            "actual_seq_qlen": [q_len],
            "actual_seq_kvlen": [kv_len],
            "atten_mask": mask,
            "sparse_mode": 3,
            "pre_tokens": 2147483647,
            "next_tokens": 0,
            "return_softmax_lse": True,
        }
        return query, key, value, cache, slots, kwargs

    @staticmethod
    def quantize(key, value):
        count = key.shape[0]
        k8, ks = torch_npu.npu_dynamic_quant(
            key.reshape(count, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        v8, vs = torch_npu.npu_dynamic_quant(
            value.reshape(count, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        return k8.view_as(key), v8.view_as(value), ks, vs

    @staticmethod
    def golden(query, quantized):
        k8, v8, ks, vs = (x.cpu() for x in quantized)
        k = (k8.float() * ks[:, None, None]).bfloat16().float()
        v = (v8.float() * vs[:, None, None]).bfloat16().float()
        q = query.cpu().float()
        ratio = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
        scores = torch.einsum("thd,shd->hts", q, k) * q.shape[-1] ** -0.5
        q_len, kv_len = q.shape[0], k.shape[0]
        mask = torch.arange(kv_len)[None] > torch.arange(q_len)[:, None] + kv_len - q_len
        scores.masked_fill_(mask[None], float("-inf"))
        return torch.einsum("hts,shd->thd", scores.softmax(-1), v).bfloat16(), scores.logsumexp(-1).t()[..., None]

    def check_cache(self, cache, slots, quantized):
        positions = slots.cpu()
        pages, offsets = positions // 512, positions % 512
        for tensor, expected in zip(cache, quantized):
            actual = tensor.cpu()
            if tensor.dtype == torch.float8_e4m3fn:
                # Byte comparison checks scatter bit-for-bit, including signed zero.
                actual = actual.view(torch.uint8)[pages, offsets]
                expected = expected.cpu().reshape(expected.shape[0], -1).view(torch.uint8)
            else:
                actual, expected = actual[pages, offsets], expected.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def check_attention(self, actual, lse, query, quantized):
        expected, expected_lse = self.golden(query, quantized)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.01, atol=0.01)

    def test_prefill_and_single_token_decode(self):
        for q_len in (1, 6, 128):
            for padding in (0, 64):
                with self.subTest(q_len=q_len, padding=padding):
                    q, k, v, cache, slots, kwargs = self.make_case(q_len, padding=padding)
                    quantized = self.quantize(k, v)
                    self.scatter_fia_fp8_cache(*quantized, cache, slots)
                    actual, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
                    torch.npu.synchronize()
                    self.check_cache(cache, slots, quantized)
                    self.check_attention(actual, lse, q, quantized)

    def test_graph_replay_updates_data_and_scales(self):
        q, k, v, cache, slots, kwargs = self.make_case(1)

        def forward():
            self.scatter_fia_fp8_cache(*self.quantize(k, v), cache, slots)
            return torch_npu.npu_fused_infer_attention_score_v2(**kwargs)

        for _ in range(3):
            forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual, lse = forward()
        for _ in range(2):
            k.mul_(2)
            v.mul_(0.5)
            graph.replay()
            torch.npu.synchronize()
            quantized = self.quantize(k, v)
            self.check_cache(cache, slots, quantized)
            self.check_attention(actual, lse, q, quantized)


if __name__ == "__main__":
    unittest.main()
