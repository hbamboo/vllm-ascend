# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Production MXFP8 SWA producer, exact window boundaries and graph replay."""

import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


class TestQfaSlidingNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        loaders = runpy.run_path(str(Path(__file__).with_name("test_c8_qfa_fp8.py")))
        cls.helpers, cls.native = loaders["load_qfa_helpers"]()
        loaders["load_triton_helper"](root, "mxfp_k_scale")

    def make_case(self, q_len, window, kv_len=1025):
        torch.manual_seed(42)
        heads, kv_heads, dim, block_size = 6, 2, 128, 512
        pages = (kv_len + block_size - 1) // block_size
        pitch = self.helpers.mxfp_packet_size_bytes(kv_heads, block_size, dim, dim) + 256
        raw = torch.zeros((pages + 1) * pitch, dtype=torch.int8, device="npu")
        cache = self.helpers.mxfp_paged_cache_views(
            raw, pages + 1, kv_heads, dim, dim, block_size, page_stride_bytes=pitch
        )
        codes = torch.arange(kv_heads * dim, device="npu").remainder(4).to(torch.uint8) + 125
        scales = torch.exp2(codes.float() - 127)
        self.helpers.fill_mxfp_v_scale_cache(codes, cache[3])
        query = (torch.randn(q_len, heads, dim, device="npu") / 8).bfloat16()
        key = (torch.randn(kv_len, kv_heads, dim, device="npu") / 8).bfloat16()
        value = (torch.randn(kv_len, kv_heads, dim, device="npu") / 4).bfloat16()
        layer = SimpleNamespace(v_cache_scale_float_reciprocal=scales.reciprocal().bfloat16())
        impl = self.helpers.AscendC8MXFPAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, dim
        impl.scale, impl.sliding_window = dim**-0.5, window
        impl.kv_sharing_target_layer_name = None
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=kv_len,
            num_actual_tokens=kv_len,
            query_start_loc_gpu=torch.tensor([0, kv_len], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([kv_len], dtype=torch.int32, device="npu"),
            block_tables=torch.arange(pages, 0, -1, dtype=torch.int32, device="npu")[None],
            attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1),
        )
        pos = torch.arange(kv_len, dtype=torch.int64, device="npu")
        metadata.slot_mapping = (pages - pos // block_size) * block_size + pos % block_size
        impl.forward(
            layer,
            query[:1].expand(kv_len, -1, -1).contiguous(),
            key,
            value,
            cache,
            metadata,
            output=torch.empty(kv_len, heads, dim, dtype=torch.bfloat16, device="npu"),
        )
        metadata = SimpleNamespace(**vars(metadata))
        metadata.qfa_metadata_cache = {}
        metadata.num_actual_tokens = metadata.max_query_len = q_len
        metadata.query_start_loc_gpu = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
        output = torch.empty_like(query)
        return SimpleNamespace(**locals())

    def reference(self, case):
        def dequant(x):
            payload, codes = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            scales = torch.exp2(codes.view(torch.uint8).cpu().float().flatten(-2) - 127)
            return (payload.cpu().float().reshape(*x.shape[:-1], -1, 32) * scales[..., None]).reshape(x.shape)

        q, k = dequant(case.query), dequant(case.key)
        v = (
            torch_npu.npu_quantize(
                case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
            )
            .cpu()
            .float()
            .reshape_as(case.value)
        )
        v *= case.scales.cpu().reshape(1, case.kv_heads, case.dim)
        used = int(case.metadata.seq_lens_gpu.cpu()[0])
        scores = torch.einsum("thd,shd->hts", q, k[:used].repeat_interleave(3, 1)) * case.impl.scale
        right = torch.arange(case.q_len)[:, None] + used - case.q_len
        pos = torch.arange(used)[None]
        mask = (pos > right) | (pos <= right - case.window)
        scores.masked_fill_(mask[None], float("-inf"))
        result = torch.einsum("hts,shd->thd", scores.softmax(-1), v[:used].repeat_interleave(3, 1))
        if hasattr(case.layer, "sink_key"):
            logits = (case.query.cpu().float() * case.layer.sink_key.cpu().float().repeat_interleave(3, 0)[None]).sum(
                -1
            )
            result *= torch.sigmoid(scores.logsumexp(-1).t() - logits * case.impl.scale)[..., None]
        return result.bfloat16()

    def run_case(self, case):
        case.impl.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=case.output)

    def test_decode_mtp_prefill_and_sink(self):
        for q_len, window, kv_len in ((1, 128, 1025), (6, 300, 1025), (129, 512, 1025), (6, 4096, 4609)):
            with self.subTest(q_len=q_len, window=window):
                case = self.make_case(q_len, window, kv_len)
                case.layer.sink_key = torch.randn(2, 128, device="npu", dtype=torch.bfloat16) / 8
                self.run_case(case)
                torch.npu.synchronize()
                torch.testing.assert_close(case.output.cpu(), self.reference(case), rtol=0.03, atol=0.005)
                print("SWA_PASS", q_len, window, kv_len, flush=True)

    def test_window_includes_exactly_window_tokens(self):
        case = self.make_case(1, 128)
        case.query.zero_()
        case.value.zero_()
        case.value[-129, :, 0] = 64  # Immediately outside the window.
        case.value[-128, :, 0] = 32  # First included token.
        case.value[-1, :, 0] = 16
        key8, _ = torch_npu.npu_dynamic_mx_quant(case.key, dst_type=torch.float8_e4m3fn)
        value8 = torch_npu.npu_quantize(
            case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
        ).reshape_as(case.value)
        self.helpers.scatter_mxfp_pa_nz_kv_cache(key8, value8, case.cache[0], case.cache[1], case.metadata.slot_mapping)
        self.run_case(case)
        torch.npu.synchronize()
        expected = torch.zeros_like(case.output.cpu())
        expected[:, :, 0] = 48 / 128
        torch.testing.assert_close(case.output.cpu(), expected, rtol=0.01, atol=0.002)

    def test_graph_refreshes_query_and_visible_lengths(self):
        case = self.make_case(6, 300)
        shared = {}
        readers = [SimpleNamespace(**vars(case.metadata)) for _ in range(2)]
        outputs = [torch.empty_like(case.query) for _ in readers]
        for metadata in readers:
            metadata.qfa_metadata_cache = {}
            metadata.qfa_shared_metadata_cache = shared

        def run(metadata_list, output_list):
            for metadata, output in zip(metadata_list, output_list):
                case.impl.forward(case.layer, case.query, None, None, case.cache, metadata, output=output)

        with patch.object(
            sys.modules["cann_ops_transformer.ops"],
            "quant_flash_attn_metadata",
            wraps=self.native.quant_flash_attn_metadata,
        ) as metadata_op:
            for _ in range(3):
                run(readers, outputs)
            torch.npu.synchronize()
            shared.clear()
            for metadata in readers:
                metadata.qfa_metadata_cache = {}
            metadata_op.reset_mock()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run(readers, outputs)
            self.assertEqual(metadata_op.call_count, 1)
            previous = outputs[0].clone()

            for step, used in enumerate((1017, 999), start=1):
                case.query.mul_(1.5)
                case.metadata.seq_lens_gpu.fill_(used)
                graph.replay()
                torch.npu.synchronize()
                captured = [output.clone() for output in outputs]
                self.assertEqual(metadata_op.call_count, step)
                self.assertFalse(torch.equal(captured[0], previous))
                previous = captured[0]
                eager_shared = {}
                eager_metadata = [SimpleNamespace(**vars(metadata)) for metadata in readers]
                eager_outputs = [torch.empty_like(case.query) for _ in readers]
                for metadata in eager_metadata:
                    metadata.qfa_metadata_cache = {}
                    metadata.qfa_shared_metadata_cache = eager_shared
                run(eager_metadata, eager_outputs)
                torch.npu.synchronize()
                self.assertEqual(metadata_op.call_count, step + 1)
                for actual, expected in zip(captured, eager_outputs):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(*captured, rtol=0, atol=0)
                torch.testing.assert_close(captured[0].cpu(), self.reference(case), rtol=0.03, atol=0.005)
                print("SWA_SHARED_GRAPH_PASS", used, flush=True)


if __name__ == "__main__":
    unittest.main()
