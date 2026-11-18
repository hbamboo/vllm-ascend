# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU QFA mode 6 producer, paged history, learned sink and graph replay.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. Production definitions are
loaded without initializing vLLM distributed groups. All quantizers, cache
writes and attention calls execute the installed Ascend 950 operators.
"""

import ast
import importlib.machinery
import importlib.util
import runpy
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# The sibling triton/ test directory shadows the installed compiler when
# this file is executed directly rather than through pytest.
if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


def load_qfa_helpers():
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
    namespace = vars(loader(torch_npu)).copy()
    path = root / "vllm_ascend/attention/attention_c8_qfa_fp8.py"
    definitions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        or isinstance(node, ast.ClassDef)
        and node.name == "AscendC8QFAFP8AttentionBackendImpl"
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)

    # Import the actual installed QFA wrapper without the package __init__,
    # which eagerly JIT-builds unrelated operators in a standalone process.
    package = Path(importlib.util.find_spec("cann_ops_transformer").origin).parent
    parent = types.ModuleType("cann_ops_transformer")
    parent.__path__ = [str(package)]
    parent.__file__ = str(package / "__init__.py")
    parent.__spec__ = importlib.machinery.ModuleSpec("cann_ops_transformer", loader=None, is_package=True)
    sys.modules[parent.__name__] = parent
    name = "c8_qfa_fp8_native_api"
    spec = importlib.util.spec_from_file_location(name, package / "ops/attention/quant_flash_attn/quant_flash_attn.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    ops = types.ModuleType("cann_ops_transformer.ops")
    ops.__path__ = [str(package / "ops")]
    ops.quant_flash_attn = native.quant_flash_attn
    ops.quant_flash_attn_metadata = native.quant_flash_attn_metadata
    sys.modules[ops.__name__] = ops
    return SimpleNamespace(**namespace), native


class TestC8QFAFP8NPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        cls.helpers, cls.native = load_qfa_helpers()
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_case(self, q_len, padding=0):
        torch.manual_seed(101)
        heads, dim, kv_len, kernel_block_size = 6, 128, 529, 128
        logical_block_size, logical_blocks = 512, 3
        pitch = self.helpers.mxfp_packet_size_bytes(1, logical_block_size, dim, dim) // 4 + padding
        raw = torch.full((logical_blocks * 4 * pitch,), 23, dtype=torch.int8, device="npu")
        spec = SimpleNamespace(
            num_kv_heads=1, head_size=dim, head_size_v=dim, block_size=logical_block_size, page_size_bytes=pitch * 4
        )
        cache = self.helpers.mxfp_cache_views_for_spec(raw, spec, kernel_block_size, qfa_fp8=True)
        query = (torch.randn(q_len, heads, dim) / 4).bfloat16().npu()
        magnitude = torch.exp2(torch.arange(kv_len).remainder(7).float() - 3).reshape(-1, 1, 1)
        key = (torch.randn(kv_len, 1, dim) * magnitude).bfloat16().npu()
        value = (torch.randn(kv_len, 1, dim) / 4).bfloat16().npu()
        channel_scale = torch.exp2(torch.arange(dim).remainder(5).float() - 8).reshape(1, dim).npu()
        layer = SimpleNamespace(v_cache_scale_float_reciprocal=channel_scale.reciprocal().flatten().bfloat16())
        positions = torch.arange(kv_len, device="npu", dtype=torch.int64)
        slots = torch.where(positions < logical_block_size, positions + 2 * logical_block_size, positions)
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=q_len,
            num_actual_tokens=q_len,
            actual_seq_lengths_q=[q_len],
            seq_lens_list=[kv_len],
            query_start_loc_gpu=torch.tensor([0, q_len], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([kv_len], dtype=torch.int32, device="npu"),
            block_tables=torch.tensor([[8, 9, 10, 11, 4, 5, 6, 7]], dtype=torch.int32, device="npu"),
            attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu(),
        )
        producer_metadata = SimpleNamespace(**vars(metadata))
        producer_metadata.max_query_len = producer_metadata.num_actual_tokens = kv_len
        producer_metadata.actual_seq_lengths_q = [kv_len]
        producer_metadata.query_start_loc_gpu = torch.tensor([0, kv_len], dtype=torch.int32, device="npu")
        producer_metadata.seq_lens_gpu = metadata.seq_lens_gpu.clone()
        producer_metadata.slot_mapping = slots
        producer_query = query[:1].expand(kv_len, -1, -1).contiguous()
        producer_output = torch.empty_like(producer_query)
        impl = self.helpers.AscendC8QFAFP8AttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = heads, 1, dim
        impl.scale, impl.sliding_window = dim**-0.5, None
        impl.kv_sharing_target_layer_name = None
        impl.qfa_v_descale = torch.ones(1, dtype=torch.float32, device="npu")
        impl.qfa_p_scale = torch.tensor([448.0], dtype=torch.float32, device="npu")
        impl.qfa_v_output_scale = channel_scale
        output = torch.empty_like(query)
        return SimpleNamespace(**locals())

    def run_case(self, case):
        case.impl.forward(
            case.layer,
            case.producer_query,
            case.key,
            case.value,
            case.cache,
            case.producer_metadata,
            output=case.producer_output,
        )
        case.impl.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=case.output)

    def reference(self, case):
        q8, qs = torch_npu.npu_dynamic_quant(case.query.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        k8, ks = torch_npu.npu_dynamic_quant(case.key.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        v8 = torch_npu.npu_quantize(
            case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
        )
        q8, qs, k8, ks, v8 = (x.cpu() for x in (q8, qs, k8, ks, v8))
        q = q8.reshape(case.q_len, case.heads, case.dim).float() * qs[:, None, None]
        k = k8.reshape(529, 1, case.dim).float() * ks[:, None, None]
        v = v8.reshape(529, 1, case.dim).float() * case.channel_scale.cpu()[None]
        scores = torch.einsum("thd,shd->hts", q, k[: case.kv_len].repeat_interleave(case.heads, 1)) * case.impl.scale
        mask = torch.arange(case.kv_len)[None] > torch.arange(case.q_len)[:, None] + case.kv_len - case.q_len
        scores.masked_fill_(mask[None], float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v[: case.kv_len].repeat_interleave(case.heads, 1))
        if hasattr(case.layer, "sink_key"):
            sink_logits = (q * case.layer.sink_key.cpu()).sum(-1) * case.impl.scale
            expected *= torch.sigmoid(scores.logsumexp(-1).t() - sink_logits)[..., None]
        return expected, scores, (q8, qs, k8, ks, v8)

    def check_case(self, case):
        expected, _, (_, _, k8, ks, v8) = self.reference(case)
        pages, offsets = case.slots.cpu() // 128, case.slots.cpu() % 128
        for cached, source in zip(case.cache[:2], (k8, v8)):
            gathered = cached.cpu().view(torch.uint8)[pages, :, offsets, :].reshape(529, -1)
            torch.testing.assert_close(gathered, source.view(torch.uint8), rtol=0, atol=0)
        cached_scales = case.cache[2].cpu()[pages, :, offsets]
        torch.testing.assert_close(cached_scales, ks[:, None], rtol=0, atol=0)
        self.assertGreater(ks.unique().numel(), 1)
        payload_bytes = 2 * 128 * 128 + 128 * 4
        self.assertTrue(torch.all(case.raw.cpu().reshape(-1, case.pitch)[:, payload_bytes:] == 23))
        torch.testing.assert_close(case.output.cpu(), expected.bfloat16(), rtol=0.03, atol=0.03)
        print(
            "QFA_FP8_PASS",
            "q_len",
            case.q_len,
            "kv_len",
            case.kv_len,
            "padding",
            case.padding,
            "max_error",
            (case.output.cpu().float() - expected).abs().max().item(),
            flush=True,
        )

    def test_prefill_decode_and_page_padding(self):
        for q_len in (1, 6, 129):
            for padding in (0, 256):
                with self.subTest(q_len=q_len, padding=padding):
                    case = self.make_case(q_len, padding)
                    self.run_case(case)
                    torch.npu.synchronize()
                    self.check_case(case)

    def test_lse_and_learned_key_sink_use_quantized_query(self):
        for q_len in (1, 6):
            with self.subTest(q_len=q_len):
                case = self.make_case(q_len)
                _, scores, (q8, qs, _, _, _) = self.reference(case)
                q_hat = q8.reshape(q_len, 6, 128).float() * qs[:, None, None]
                direction = q_hat[0, 0]
                delta = case.query.cpu()[0, 0].float() - direction
                orthogonal = delta - direction * delta.dot(direction) / direction.square().sum()
                # The selected head has factor ~0.5. A component orthogonal
                # to Qhat makes use of raw Q change its sink logit by ~2,
                # so this also detects an inconsistent sink query producer.
                sink = direction * scores[0, 0].logsumexp(-1) / (direction.square().sum() * case.impl.scale)
                sink += orthogonal * 2 / (orthogonal.square().sum() * case.impl.scale)
                case.layer.sink_key = sink[None].npu()
                self.run_case(case)
                self.check_case(case)
                expected, scores, (q8, qs, _, _, _) = self.reference(case)
                relative_error = (case.output.cpu()[0, 0].float() - expected[0, 0]).norm() / expected[0, 0].norm()
                self.assertLess(relative_error.item(), 0.08)
                q_ntd = q8.reshape(q_len, 6, 128).view(torch.uint8).permute(1, 0, 2).contiguous().npu()
                q_ntd = q_ntd.view(torch.float8_e4m3fn)
                mask_mode = 0 if q_len == 1 else 3
                plans = case.metadata.qfa_metadata_cache
                key = ("fp8_token", 6, 1, 128, q_len, mask_mode)
                _, lse = self.native.quant_flash_attn(
                    q_ntd,
                    case.cache[0],
                    case.cache[1],
                    qs[None].expand(6, -1).contiguous().npu(),
                    case.cache[2],
                    case.impl.qfa_v_descale,
                    6,
                    p_scale=case.impl.qfa_p_scale,
                    block_table=case.metadata.block_tables,
                    cu_seqlens_q=case.metadata.query_start_loc_gpu,
                    seqused_kv=case.metadata.seq_lens_gpu,
                    metadata=plans[key],
                    softmax_scale=case.impl.scale,
                    mask_mode=mask_mode,
                    attn_mask=case.metadata.attn_mask if mask_mode else None,
                    max_seqlen_q=q_len,
                    max_seqlen_kv=-1,
                    layout_q="NTD",
                    layout_q_descale="NT",
                    layout_kv="PA_BNBD",
                    layout_out="TND",
                    return_softmax_lse=True,
                )
                torch.testing.assert_close(lse.cpu().reshape(6, q_len), scores.logsumexp(-1), rtol=1e-4, atol=1e-4)

    def test_graph_replay_refreshes_lengths_and_dynamic_scales(self):
        case = self.make_case(1, 256)
        for _ in range(3):
            self.run_case(case)
        torch.npu.synchronize()
        case.metadata.qfa_metadata_cache = {}
        case.producer_metadata.qfa_metadata_cache = {}
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.run_case(case)
        for factor, kv_len in ((2.0, 513), (0.125, 529)):
            case.key.mul_(factor)
            case.query.mul_(1 / factor)
            case.kv_len = kv_len
            case.metadata.seq_lens_gpu.fill_(kv_len)
            graph.replay()
            torch.npu.synchronize()
            self.check_case(case)
            graph_output = case.output.clone()
            self.run_case(case)
            torch.npu.synchronize()
            torch.testing.assert_close(case.output, graph_output, rtol=0, atol=0)

    def test_key_extrema_are_not_clipped_to_mx_power_of_two_range(self):
        case = self.make_case(1)
        row = torch.tensor([1.875, 1.0, 0.5, -1.875], dtype=torch.bfloat16, device="npu").repeat(32)
        case.key.copy_(row.reshape(1, 1, 128))
        self.run_case(case)
        torch.npu.synchronize()
        reconstructed = case.cache[0].cpu()[8, 0, 0].float() * case.cache[2].cpu()[8, 0, 0]
        torch.testing.assert_close(reconstructed[[0, 3]].bfloat16(), row.cpu()[[0, 3]], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
