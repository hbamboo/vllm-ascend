# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU per-token K, checkpoint V scales, paged QFA and graph replay.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. The test loads production
helper/backend definitions without initializing vLLM distributed groups.
All quantization, cache writes and QFA calls execute real NPU operators.
"""

import importlib.util
import math
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The sibling triton/ test directory shadows the installed compiler when
# this file is executed directly rather than through pytest.
if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


class TestC8KPerTokenNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
        cls.helpers = loader(torch_npu)
        # Keep the real Triton compiler/runtime while avoiding vLLM platform
        # discovery in this standalone process without distributed groups.
        import triton
        import triton.language as tl

        name = "vllm_ascend.ops.triton.mxfp_k_scale"
        spec = importlib.util.spec_from_file_location(name, root / "vllm_ascend/ops/triton/mxfp_k_scale.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl)}):
            spec.loader.exec_module(module)
        sys.modules[name] = module
        from safetensors import safe_open

        checkpoint = Path("/mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8")
        cls.v_scale = None
        for path in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                name = "model.layers.1.self_attn.fa_v.scale"
                if name in set(handle.keys()):
                    cls.v_scale = handle.get_tensor(name).flatten()[:128].clone()
                    print("CHECKPOINT", str(path), name, cls.v_scale.tolist(), flush=True)
                    break
        if cls.v_scale is None:
            raise RuntimeError("Target checkpoint V scale was not found.")
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_case(self, q_len, padding=0):
        torch.manual_seed(101)
        heads, dim, kv_len, block_size = 6, 128, 529, 512
        pitch = self.helpers.mxfp_packet_size_bytes(1, block_size, dim, dim) + padding
        raw = torch.zeros(3 * pitch, dtype=torch.int8, device="npu")
        cache = self.helpers.mxfp_paged_cache_views(raw, 3, 1, dim, dim, block_size, page_stride_bytes=pitch)
        query = (torch.randn(q_len, heads, dim) / 4).bfloat16().npu()
        magnitude = torch.exp2(torch.arange(kv_len).remainder(7).float() - 3).reshape(-1, 1, 1)
        key = (torch.randn(kv_len, 1, dim) * magnitude).bfloat16().npu()
        value = (torch.randn(kv_len, 1, dim) / 4).bfloat16().npu()
        vs = self.v_scale.npu()
        self.helpers.fill_mxfp_v_scale_cache(vs, cache[3])
        reciprocal = torch.exp2(127 - vs.float()).bfloat16()
        positions = torch.arange(kv_len, device="npu", dtype=torch.int64)
        slots = torch.where(positions < block_size, positions + 2 * block_size, positions)
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=q_len,
            num_actual_tokens=q_len,
            query_start_loc_gpu=torch.tensor([0, q_len], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([kv_len], dtype=torch.int32, device="npu"),
            block_tables=torch.tensor([[2, 1]], dtype=torch.int32, device="npu"),
            attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu(),
        )
        impl = self.helpers.AscendC8MXFPAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = heads, 1, dim
        impl.scale, impl.sliding_window = dim**-0.5, None
        output = torch.empty_like(query)
        return SimpleNamespace(**locals())

    def run_case(self, case):
        q8, qs = torch_npu.npu_dynamic_mx_quant(case.query, dst_type=torch.float8_e4m3fn)
        k8, ks = self.helpers.quantize_mxfp_per_token(case.key)
        v8 = torch_npu.npu_quantize(case.value.flatten(1), case.reciprocal, None, torch.float8_e4m3fn, -1, False)
        v8 = v8.reshape_as(case.value)
        self.helpers.scatter_mxfp_pa_nz_kv_cache(k8, v8, *case.cache[:2], case.slots)
        self.helpers.scatter_mxfp_k_scale_cache(
            ks, case.cache[2], self.helpers.mxfp_k_scale_slot_index(case.slots, 512)
        )
        case.impl._forward_mxfp8_attention(
            q8, qs, case.cache, case.metadata, case.output, sink_logits=getattr(case, "sink_logits", None)
        )
        return q8, qs, k8, ks, v8

    def check_case(self, case, quantized):
        q8, qs, k8, ks, v8 = (x.cpu() for x in quantized)
        # Exactly one scale per token, repeated into the QFA group slots.
        torch.testing.assert_close(ks.flatten(1), ks[:, 0, 0, :1].expand(ks.shape[0], 4), rtol=0, atol=0)
        pages, offsets = case.slots.cpu() // 512, case.slots.cpu() % 512
        key_cache, value_cache, key_scales, value_scales = (x.cpu() for x in case.cache)
        for cached, source in ((key_cache, k8), (value_cache, v8)):
            gathered = cached.view(torch.uint8)[pages, :, :, offsets, :].reshape_as(source)
            torch.testing.assert_close(gathered, source.view(torch.uint8), rtol=0, atol=0)
        gathered_scales = key_scales[pages, :, offsets // 16, :, offsets % 16, :]
        torch.testing.assert_close(gathered_scales, ks, rtol=0, atol=0)
        self.assertTrue(torch.equal(value_scales[1, :, :, 0, :, 0].flatten(), self.v_scale))
        q = q8.float() * torch.exp2(qs.view(torch.uint8).float() - 127).repeat_interleave(32, -1).reshape_as(q8)
        k = k8.float() * torch.exp2(ks[:, 0, 0, 0].float() - 127)[:, None, None]
        v = v8.float() * torch.exp2(self.v_scale.float() - 127)[None, None, :]
        k, v = k[: case.kv_len].repeat_interleave(6, 1), v[: case.kv_len].repeat_interleave(6, 1)
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(128)
        mask = torch.arange(case.kv_len)[None, :] > torch.arange(case.q_len)[:, None] + case.kv_len - case.q_len
        scores.masked_fill_(mask[None], float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v).bfloat16()
        if hasattr(case, "sink_logits"):
            factor = torch.sigmoid(scores.logsumexp(-1).t() - case.sink_logits.cpu())
            expected = (expected.float() * factor[..., None]).bfloat16()
        actual = case.output.cpu()
        # QFA also quantizes the softmax/P path, unlike this FP32 reference.
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        print(
            "PASS",
            "q_len",
            case.q_len,
            "padding",
            case.padding,
            "max_error",
            (actual - expected).abs().max().item(),
            flush=True,
        )

    def test_prefill_decode_and_page_padding(self):
        for q_len in (1, 6, 128):
            for padding in (0, 256):
                with self.subTest(q_len=q_len, padding=padding):
                    case = self.make_case(q_len, padding)
                    quantized = self.run_case(case)
                    torch.npu.synchronize()
                    self.check_case(case, quantized)

    def test_graph_replay_refreshes_token_scales(self):
        case = self.make_case(1, 256)
        for _ in range(3):
            self.run_case(case)
        torch.npu.synchronize()
        # Capture the device-side length derivation and metadata op too.
        case.metadata.qfa_metadata_cache = {}
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            quantized = self.run_case(case)
        for factor, kv_len in ((2, 513), (0.25, 529)):
            case.key.mul_(factor)
            case.kv_len = kv_len
            case.metadata.seq_lens_gpu.fill_(kv_len)
            graph.replay()
            torch.npu.synchronize()
            self.check_case(case, quantized)
            graph_output = case.output.clone()
            self.run_case(case)
            torch.npu.synchronize()
            torch.testing.assert_close(case.output, graph_output, rtol=0, atol=0)

    def test_learned_key_sink(self):
        for q_len in (1, 6):
            with self.subTest(q_len=q_len):
                case = self.make_case(q_len)
                sink_key = torch.randn(128, device="npu", dtype=torch.float32) / 4
                case.sink_logits = (case.query.float() * sink_key).sum(-1) / math.sqrt(128)
                quantized = self.run_case(case)
                torch.npu.synchronize()
                self.check_case(case, quantized)


class TestC8KVPerTokenNPU(TestC8KPerTokenNPU):
    """Same pages and prompts, with production FIA and two dynamic scales."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import triton
        import triton.language as tl

        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        spec = importlib.util.spec_from_file_location("c8_sink", root / "vllm_ascend/ops/triton/sink.py")
        sink = importlib.util.module_from_spec(spec)
        # The sink kernel uses no cached device-property values; bypass only
        # that initialization to avoid importing vLLM platform plugins here.
        with patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=lambda: None),
            },
        ):
            spec.loader.exec_module(sink)
        cls.helpers.AscendC8FIAPseudoAttentionBackendImpl.forward.__globals__["sink_apply_triton"] = (
            sink.sink_apply_triton
        )

    def make_case(self, q_len, padding=0):
        case = super().make_case(q_len, padding)
        spec = SimpleNamespace(num_kv_heads=1, head_size=128, head_size_v=128)
        case.cache = self.helpers.fia_pseudo_cache_views(case.raw, 3, spec, 512, page_stride_bytes=case.pitch)
        case.value.mul_(torch.exp2(torch.arange(529, device="npu").remainder(5).float() - 2)[:, None, None])
        case.impl = self.helpers.AscendC8FIAPseudoAttentionBackendImpl()
        case.impl.num_heads, case.impl.num_kv_heads, case.impl.head_size, case.impl.scale = 6, 1, 128, 128**-0.5
        case.impl.kv_sharing_target_layer_name = None
        case.metadata.actual_seq_lengths_q = [q_len]
        case.metadata.seq_lens_list = [529]
        case.layer = SimpleNamespace(layer_name="attn")
        return case

    def run_case(self, case):
        k8, ks = self.helpers.quantize_mxfp_per_token(case.key)
        v8, vs = self.helpers.quantize_mxfp_per_token(case.value)
        ks = torch.exp2(ks[:, 0, 0, 0].float() - 127)
        vs = torch.exp2(vs[:, 0, 0, 0].float() - 127)
        self.helpers.scatter_fia_fp8_cache(k8, v8, ks, vs, case.cache, case.slots)
        case.impl.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=case.output)
        return k8, ks, v8, vs

    def check_case(self, case, quantized):
        k8, ks, v8, vs = (x.cpu() for x in quantized)
        pages, offsets = case.slots.cpu() // 512, case.slots.cpu() % 512
        for cached, source in zip(case.cache, (k8.flatten(1), v8.flatten(1), ks, vs)):
            cached = cached.cpu()
            if source.dtype == torch.float8_e4m3fn:
                cached, source = cached.view(torch.uint8), source.view(torch.uint8)
            torch.testing.assert_close(cached[pages, offsets], source, rtol=0, atol=0)
        self.assertGreater(vs.unique().numel(), 1)
        k = (k8.float() * ks[:, None, None]).bfloat16().float().repeat_interleave(6, 1)
        v = (v8.float() * vs[:, None, None]).bfloat16().float().repeat_interleave(6, 1)
        scores = torch.einsum("thd,shd->hts", case.query.cpu().float(), k) / math.sqrt(128)
        mask = torch.arange(529)[None] > torch.arange(case.q_len)[:, None] + 529 - case.q_len
        scores.masked_fill_(mask[None], float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v)
        if hasattr(case.layer, "sink_key"):
            logits = (case.query.cpu().float() * case.layer.sink_key.cpu()).sum(-1) / math.sqrt(128)
            expected *= torch.sigmoid(scores.logsumexp(-1).t() - logits)[..., None]
        actual = case.output.cpu()
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0.02, atol=0.02)
        print(
            "FIA_PASS",
            case.q_len,
            case.padding,
            "max_error",
            (actual - expected.bfloat16()).abs().max().item(),
            flush=True,
        )

    def test_graph_replay_refreshes_token_scales(self):
        case = self.make_case(1, 256)
        for _ in range(3):
            self.run_case(case)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            quantized = self.run_case(case)
        for factor in (2, 0.25):
            case.key.mul_(factor)
            case.value.mul_(1 / factor)
            graph.replay()
            torch.npu.synchronize()
            self.check_case(case, quantized)
            graph_output = case.output.clone()
            self.run_case(case)
            torch.npu.synchronize()
            torch.testing.assert_close(case.output, graph_output, rtol=0, atol=0)

    def test_learned_key_sink(self):
        for q_len in (1, 6):
            case = self.make_case(q_len)
            case.layer.sink_key = torch.randn(1, 128, device="npu") / 4
            quantized = self.run_case(case)
            torch.npu.synchronize()
            self.check_case(case, quantized)


if __name__ == "__main__":
    unittest.main()
