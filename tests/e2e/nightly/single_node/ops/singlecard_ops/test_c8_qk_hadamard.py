# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone NPU H128 + production C8 quantization/QFA/sink/graph regression.

Run this file directly on one free Ascend 950. No model, weights, distributed
groups, offset mode or BF16 attention fallback is used. Existing C8 fixtures
provide the real packet writer and installed native QFA wrapper.
"""

import importlib.util
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


class TestC8QKHadamardNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import triton
        import triton.language as tl

        torch.npu.set_device(0)
        cls.root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        full = runpy.run_path(str(Path(__file__).with_name("test_c8_qfa_fp8.py")))
        swa = runpy.run_path(str(Path(__file__).with_name("test_c8_qfa_sliding.py")))
        cls.helpers, cls.native = full["load_qfa_helpers"]()
        full["load_triton_helper"](cls.root, "mxfp_k_scale")
        props = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        vector_cores = props["num_vectorcore"]
        assert vector_cores > 0
        name = "vllm_ascend.ops.triton.qk_hadamard"
        spec = importlib.util.spec_from_file_location(name, cls.root / "vllm_ascend/ops/triton/qk_hadamard.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(
                    init_device_properties_triton=lambda: None, get_vectorcore_num=lambda: vector_cores
                ),
            },
        ):
            spec.loader.exec_module(module)
        cls.rotate = staticmethod(module.hadamard128)
        cls.full_fixture = full["TestC8QFAFP8NPU"]()
        cls.swa_fixture = swa["TestQfaSlidingNPU"]()
        for fixture in (cls.full_fixture, cls.swa_fixture):
            fixture.helpers, fixture.native = cls.helpers, cls.native
        definitions = runpy.run_path(str(cls.root / "tests/ut/attention/test_c8_qk_hadamard.py"))
        cls.h128 = torch.tensor(definitions["dense_hadamard"](), dtype=torch.float64)
        print("H128_NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def test_fht_strides_dtype_and_independent_dense_reference(self):
        torch.manual_seed(20261121)
        fixtures = [
            torch.eye(128, dtype=torch.bfloat16, device="npu"),
            torch.randn(14, 6, 256, dtype=torch.bfloat16, device="npu")[::2, :, ::2],
            torch.randn(129, 2, 128, dtype=torch.bfloat16, device="npu"),
            torch.zeros(0, 6, 128, dtype=torch.bfloat16, device="npu"),
        ]
        for source in fixtures:
            before = source.cpu().clone()
            expected = before.double() @ self.h128
            fp32 = self.rotate(source, output_dtype=torch.float32)
            bf16 = self.rotate(source)
            torch.npu.synchronize()
            self.assertEqual(fp32.dtype, torch.float32)
            self.assertEqual(bf16.dtype, torch.bfloat16)
            torch.testing.assert_close(source.cpu(), before, rtol=0, atol=0)
            torch.testing.assert_close(fp32.cpu().double(), expected, rtol=2e-6, atol=2e-6)
            # One final BF16 rounding; near a tie the FP32 butterfly can differ
            # from the FP64 oracle by one BF16 ULP.
            torch.testing.assert_close(bf16.cpu().float(), expected.bfloat16().float(), rtol=0.008, atol=2e-5)
            print("FHT_PASS", tuple(source.shape), tuple(source.stride()), flush=True)

    def make_case(self, sliding):
        torch.manual_seed(20261122)
        if sliding:
            with patch.object(self.helpers.AscendC8MXFPAttentionBackendImpl, "c8_qk_rotation", "hadamard"):
                case = self.swa_fixture.make_case(6, 300, kv_len=513)
        else:
            case = self.full_fixture.make_case(6, kv_heads=2, kv_len=257)
            case.impl.q_quant_mode = case.impl.k_quant_mode = "per_token_head"
            case.impl.v_quant_mode = "per_channel"
            case.channel_scale = torch.exp2(torch.arange(256, device="npu").remainder(4).float() - 1).reshape(2, 128)
            case.layer.v_cache_scale_float_reciprocal = case.channel_scale.reciprocal().flatten().bfloat16()
            case.impl.qfa_v_descale.fill_(1)
            case.impl.qfa_v_output_scale = case.channel_scale
        case.impl.c8_qk_rotation = "hadamard"
        case.layer.sink_key = (torch.randn(2, 128, device="npu") / 8).bfloat16()
        original_sink = case.layer.sink_key.clone()
        case.layer._c8_rotated_sink_key = self.rotate(case.layer.sink_key, output_dtype=torch.float32)
        torch.testing.assert_close(case.layer.sink_key, original_sink, rtol=0, atol=0)
        if not sliding:
            self.full_fixture.run_case(case)
        return case

    def golden(self, case, sliding):
        # Check FHT independently first. The attention golden then uses actual
        # quantizer codes/scales, isolating native QFA/P448 from FHT rounding.
        q_rot, k_rot = self.rotate(case.query), self.rotate(case.key)
        for actual, original in ((q_rot, case.query), (k_rot, case.key)):
            expected = (original.cpu().double() @ self.h128).bfloat16().float()
            torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.008, atol=2e-5)
        if sliding:
            q8, qs = torch_npu.npu_dynamic_mx_quant(q_rot, dst_type=torch.float8_e4m3fn)
            k8, ks = torch_npu.npu_dynamic_mx_quant(k_rot, dst_type=torch.float8_e4m3fn)

            def decode(payload, scales):
                decoded = torch.exp2(scales.view(torch.uint8).cpu().double().flatten(-2) - 127)
                return (payload.cpu().double().reshape(*payload.shape[:-1], 4, 32) * decoded[..., None]).flatten(-2)

            qhat, khat = decode(q8, qs), decode(k8, ks)
            vscale = case.scales.cpu().double().reshape(2, 128)
        else:
            q8, qs = torch_npu.npu_dynamic_quant(q_rot, dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
            k8, ks = torch_npu.npu_dynamic_quant(k_rot, dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
            qhat, khat = (
                q8.cpu().double() * qs.cpu().double()[..., None],
                k8.cpu().double() * ks.cpu().double()[..., None],
            )
            vscale = case.channel_scale.cpu().double()
        v8 = torch_npu.npu_quantize(
            case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
        ).reshape_as(case.value)
        vhat = v8.cpu().double() * vscale[None]
        used = case.kv_len
        group = case.heads // case.kv_heads
        scores = torch.einsum("thd,shd->hts", qhat, khat[:used].repeat_interleave(group, 1)) * case.impl.scale
        right = torch.arange(case.q_len)[:, None] + used - case.q_len
        positions = torch.arange(used)[None]
        mask = positions > right
        if sliding:
            mask |= positions <= right - case.window
        scores.masked_fill_(mask[None], float("-inf"))
        lse = scores.logsumexp(-1)
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), vhat[:used].repeat_interleave(group, 1))
        sink_q = q_rot.cpu().double() if sliding else qhat
        sink = case.layer._c8_rotated_sink_key.cpu().double().repeat_interleave(group, 0)
        sink_logits = (sink_q * sink[None]).sum(-1) * case.impl.scale
        expected *= torch.sigmoid(lse.t() - sink_logits)[..., None]
        return expected, lse, k8

    def check_cached_k(self, case, sliding, k8):
        cached = case.cache[0].view(torch.uint8).cpu()
        table = case.metadata.block_tables[0].cpu().tolist()
        block_size = 512 if sliding else 128
        if sliding:
            actual = torch.stack(
                [cached[table[t // block_size], :, :, t % block_size, :].reshape(2, 128) for t in range(case.kv_len)]
            )
        else:
            actual = torch.stack([cached[table[t // block_size], :, t % block_size, :] for t in range(case.kv_len)])
        torch.testing.assert_close(actual, k8.view(torch.uint8).cpu(), rtol=0, atol=0)

    def test_full_swa_quantized_scores_sink_and_graph(self):
        for sliding in (False, True):
            with self.subTest(sliding=sliding):
                case = self.make_case(sliding)
                ops = sys.modules["cann_ops_transformer.ops"]
                native_op, captured = ops.quant_flash_attn, []

                def record(*args, _native_op=native_op, _captured=captured, **kwargs):
                    result = _native_op(*args, **kwargs)
                    _captured.append(result)
                    return result

                def run(output=case.output, *, _case=case):
                    return _case.impl.forward(
                        _case.layer, _case.query, None, None, _case.cache, _case.metadata, output=output
                    )

                with patch.object(ops, "quant_flash_attn", side_effect=record):
                    for _ in range(3):
                        run()
                    torch.npu.synchronize()
                    expected, lse, k8 = self.golden(case, sliding)
                    self.check_cached_k(case, sliding, k8)
                    torch.testing.assert_close(captured[-1][1].cpu().double(), lse, rtol=1e-4, atol=1e-3)
                    torch.testing.assert_close(case.output.cpu().double(), expected, rtol=0.03, atol=0.005)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        run()
                    graph_lse = captured[-1][1]
                    for factor in (1.25, -0.75):
                        case.query.mul_(factor)
                        graph.replay()
                        torch.npu.synchronize()
                        graph_output, replay_lse = case.output.clone(), graph_lse.clone()
                        eager = torch.empty_like(case.output)
                        run(eager)
                        torch.npu.synchronize()
                        torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)
                        torch.testing.assert_close(replay_lse, captured[-1][1], rtol=0, atol=0)
                        expected, lse, _ = self.golden(case, sliding)
                        torch.testing.assert_close(replay_lse.cpu().double(), lse, rtol=1e-4, atol=1e-3)
                        torch.testing.assert_close(graph_output.cpu().double(), expected, rtol=0.03, atol=0.005)
                print("H128_QUANT_QFA_GRAPH_PASS", "SWA" if sliding else "FULL", flush=True)


if __name__ == "__main__":
    unittest.main()
