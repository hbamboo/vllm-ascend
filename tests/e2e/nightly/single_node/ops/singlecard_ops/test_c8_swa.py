# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU sliding-window C8 attention: FIA v2 BAND over the dense FP8 page.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. The production
``AscendC8FIAPseudoAttentionBackendImpl`` is loaded without initializing vLLM
distributed groups; quantization, cache writes and FIA calls are real NPU
operators.

The two references are:
  * an FP32 golden built from the *quantized* K/V with the exact sliding
    window (operator/paged-cache correctness, quantization excluded);
  * a BF16 FIA v2 call with the same window (the C16 semantics the C8 path
    must match, quantization included).
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

HEADS, DIM, KV_HEADS = 6, 128, 1
BLOCK_SIZE = 512
KV_LEN = 1500
WINDOW = 512
SCALE = DIM**-0.5


class TestC8SWANPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
        cls.helpers = loader(torch_npu)
        # Real Triton sink kernel, without vLLM platform discovery.
        import triton
        import triton.language as tl

        spec = importlib.util.spec_from_file_location("c8_swa_sink", root / "vllm_ascend/ops/triton/sink.py")
        sink = importlib.util.module_from_spec(spec)
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
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_case(self, q_len, window=WINDOW, kv_len=KV_LEN, sink=False):
        torch.manual_seed(101)
        num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        spec = SimpleNamespace(num_kv_heads=KV_HEADS, head_size=DIM, head_size_v=DIM)
        pitch = self.helpers.mxfp_packet_size_bytes(KV_HEADS, BLOCK_SIZE, DIM, DIM)
        raw = torch.zeros(num_blocks * pitch, dtype=torch.int8, device="npu")
        cache = self.helpers.fia_pseudo_cache_views(raw, num_blocks, spec, BLOCK_SIZE, page_stride_bytes=pitch)
        # Varying magnitudes across tokens exercise the per-token scales.
        magnitude = torch.exp2(torch.arange(kv_len).remainder(7).float() - 3).reshape(-1, 1, 1)
        key = (torch.randn(kv_len, KV_HEADS, DIM) * magnitude).bfloat16().npu()
        value = (torch.randn(kv_len, KV_HEADS, DIM) / 4).bfloat16().npu()
        query = (torch.randn(q_len, HEADS, DIM) / 4).bfloat16().npu()
        slots = torch.arange(kv_len, dtype=torch.int64, device="npu")
        block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
        impl = self.helpers.AscendC8FIAPseudoAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = HEADS, KV_HEADS, DIM
        impl.scale, impl.sliding_window = SCALE, window
        impl.kv_sharing_target_layer_name = None
        layer = SimpleNamespace(layer_name="model.layers.1.self_attn.attn")
        if sink:
            layer.sink_key = torch.randn(KV_HEADS, DIM, device="npu") / 4
        output = torch.empty_like(query)
        # Producer call: write the whole history through the production path.
        producer_metadata = SimpleNamespace(
            causal=True,
            max_query_len=kv_len,
            num_actual_tokens=kv_len,
            actual_seq_lengths_q=[kv_len],
            seq_lens_list=[kv_len],
            slot_mapping=slots,
            block_tables=block_table,
            attn_mask=mask,
        )
        producer_query = query[:1].expand(kv_len, -1, -1).contiguous()
        producer_output = torch.empty_like(producer_query)
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=q_len,
            num_actual_tokens=q_len,
            actual_seq_lengths_q=[q_len],
            seq_lens_list=[kv_len],
            slot_mapping=slots[:1],
            block_tables=block_table,
            attn_mask=mask,
        )
        return SimpleNamespace(
            q_len=q_len,
            window=window,
            kv_len=kv_len,
            key=key,
            value=value,
            query=query,
            cache=cache,
            impl=impl,
            layer=layer,
            output=output,
            producer_query=producer_query,
            producer_output=producer_output,
            producer_metadata=producer_metadata,
            metadata=metadata,
            block_table=block_table,
            mask=mask,
            block_size=BLOCK_SIZE,
        )

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
        torch.npu.synchronize()
        return case.output

    @staticmethod
    def dequantized(case):
        k8, ks = torch_npu.npu_dynamic_quant(
            case.key.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        v8, vs = torch_npu.npu_dynamic_quant(
            case.value.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        # BF16 rounding mirrors the cache payload the operator reads back.
        k = (k8.float() * ks[:, None]).bfloat16().float()
        v = (v8.float() * vs[:, None]).bfloat16().float()
        return k, v

    def golden(self, case, windowed):
        k, v = self.dequantized(case)
        q = case.query.cpu().float()
        k, v = k.cpu(), v.cpu()
        out = torch.zeros(case.q_len, HEADS, DIM)
        logsumexp = torch.zeros(case.q_len, HEADS)
        for i in range(case.q_len):
            pos = case.kv_len - case.q_len + i
            start = max(0, pos - case.window + 1) if windowed else 0
            logits = torch.einsum("hd,sd->hs", q[i], k[start : pos + 1]) * SCALE
            logsumexp[i] = torch.logsumexp(logits.float(), dim=-1)
            out[i] = torch.einsum("hs,sd->hd", logits.softmax(-1), v[start : pos + 1])
        if hasattr(case.layer, "sink_key"):
            sink_logits = (q * case.layer.sink_key.cpu()).sum(-1) * SCALE
            out *= torch.sigmoid(logsumexp - sink_logits)[..., None]
        return out

    def bf16_reference(self, case):
        """Same window through the BF16 FIA path (C16 semantics)."""
        cache_k = torch.zeros(
            (case.kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE,
            BLOCK_SIZE,
            KV_HEADS * DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        cache_v = torch.zeros_like(cache_k)
        pages = torch.arange(case.kv_len, device="npu") // BLOCK_SIZE
        offsets = torch.arange(case.kv_len, device="npu") % BLOCK_SIZE
        cache_k[pages, offsets] = case.key.reshape(case.kv_len, -1)
        cache_v[pages, offsets] = case.value.reshape(case.kv_len, -1)
        out, lse = torch_npu.npu_fused_infer_attention_score_v2(
            query=case.query,
            key=cache_k,
            value=cache_v,
            num_query_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="TND",
            block_table=case.block_table,
            block_size=BLOCK_SIZE,
            actual_seq_qlen=[case.q_len],
            actual_seq_kvlen=[case.kv_len],
            atten_mask=case.mask,
            sparse_mode=4,
            pre_tokens=case.window,
            next_tokens=0,
            softmax_scale=SCALE,
            return_softmax_lse=True,
        )
        torch.npu.synchronize()
        out = out.float().reshape(case.q_len, HEADS, DIM).cpu()
        if hasattr(case.layer, "sink_key"):
            sink_logits = (case.query.cpu().float() * case.layer.sink_key.cpu()).sum(-1) * SCALE
            out *= torch.sigmoid(lse.cpu().float().reshape(case.q_len, HEADS) - sink_logits)[..., None]
        return out

    def check_case(self, case, label):
        actual = self.run_case(case).cpu().float()
        windowed = self.golden(case, windowed=True)
        full = self.golden(case, windowed=False)
        err_window = (actual - windowed).abs().max().item()
        err_full = (actual - full).abs().max().item()
        bf16 = self.bf16_reference(case)
        err_bf16 = (actual - bf16).abs().max().item()
        print(
            f"PASS {label}: q={case.q_len} kv={case.kv_len} window={case.window} "
            f"max|err windowed golden|={err_window:.5f} full golden={err_full:.5f} "
            f"bf16={err_bf16:.5f}",
            flush=True,
        )
        # Operator/paging correctness against the quantized-KV golden.
        torch.testing.assert_close(actual, windowed, rtol=0.03, atol=0.03)
        # The window must really truncate when the context exceeds it.
        if case.kv_len > case.window and case.q_len <= case.window:
            self.assertGreater(err_full, max(err_window * 3, 0.02))
        # The C8 path must stay close to the C16/BF16 semantics.
        torch.testing.assert_close(actual, bf16, rtol=0.05, atol=0.05)

    def test_prefill_verify_band(self):
        for q_len in (8, 64):
            with self.subTest(q_len=q_len):
                self.check_case(self.make_case(q_len), f"verify q={q_len}")

    def test_decode_single_row_band(self):
        # Pure decode: FIA antiquant rejects BAND at S=1, so the impl feeds
        # two query rows and keeps the second one.
        case = self.make_case(q_len=1)
        out = self.run_case(case)
        self.assertEqual(out.shape[0], 1)
        self.check_case(case, "decode S=1")

    def test_decode_short_context_covers_all_tokens(self):
        # Context shorter than the window: the band is a no-op, full and
        # windowed goldens coincide.
        case = self.make_case(q_len=1, kv_len=300)
        actual = self.run_case(case).cpu().float()
        golden = self.golden(case, windowed=True)
        torch.testing.assert_close(actual, golden, rtol=0.03, atol=0.03)
        print(f"PASS short decode: max|err|={(actual - golden).abs().max().item():.5f}", flush=True)

    def test_window_wide_open_is_causal(self):
        # sliding_window >= context: decode must equal the causal reference.
        case = self.make_case(q_len=1, kv_len=600, window=4096)
        self.check_case(case, "wide window decode")

    def test_learned_key_sink(self):
        for q_len in (1, 8):
            with self.subTest(q_len=q_len):
                case = self.make_case(q_len=q_len, sink=True)
                self.check_case(case, f"sink q={q_len}")

    def test_graph_replay_keeps_band(self):
        case = self.make_case(q_len=1)
        self.run_case(case)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            case.impl.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=case.output)
        graph.replay()
        torch.npu.synchronize()
        replayed = case.output.clone()
        # Eager run again with the same cache must match the replay.
        case.impl.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=case.output)
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, case.output, rtol=0, atol=0)
        golden = self.golden(case, windowed=True)
        err = (replayed.cpu().float() - golden).abs().max().item()
        print(f"PASS graph replay decode: max|err|={err:.5f}", flush=True)
        torch.testing.assert_close(replayed.cpu().float(), golden, rtol=0.03, atol=0.03)


if __name__ == "__main__":
    unittest.main()
