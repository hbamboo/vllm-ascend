# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real-NPU residual quantization, causal anchor lifetime and page reassignment.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. Uses the production Triton
kernel and cache layout without initializing vLLM distributed groups.
"""

import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


class TestQfaKOffsetNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        cls.root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        cls.helpers = runpy.run_path(str(cls.root / "tests/ut/attention/test_c8_qfa_fp8.py"))["load_cache_helpers"]()
        load = runpy.run_path(str(Path(__file__).with_name("test_c8_qfa_fp8.py")))["load_triton_helper"]
        cls.quantize = staticmethod(load(cls.root, "qfa_k_offset").quantize_qfa_k_offset)

    def test_append_prefix_and_reassigned_page(self):
        torch.manual_seed(188)
        heads, dim, pitch = 2, 128, 2 * 128 * 268 + 64
        spec = SimpleNamespace(num_kv_heads=heads, head_size=dim, head_size_v=dim)
        for mode in ("per_token_head_scalar", "per_block_fp8_anchor"):
            with self.subTest(mode=mode):
                raw = torch.full((4 * pitch,), 37, dtype=torch.int8, device="npu")
                cache = self.helpers.qfa_fp8_cache_views(raw, 4, spec, 128, page_stride_bytes=pitch, k_offset_mode=mode)
                before = raw.cpu().clone()
                # Open physical page2, then append without rewriting its anchor.
                key = (torch.randn(129, heads, dim) * 2 + 3).bfloat16()
                slots = torch.arange(256, 385, dtype=torch.int64)
                saved_anchor = None
                for lo, hi in ((0, 73), (73, 129)):
                    codes, scale = self.quantize(key[lo:hi].npu(), slots[lo:hi].npu(), cache, mode)
                    torch.npu.synchronize()
                    kc = key[lo:hi].float()
                    if mode == "per_token_head_scalar":
                        offset = (kc.amax(-1, keepdim=True) + kc.amin(-1, keepdim=True)) * 0.5
                        got = torch.stack([cache[4][s // 128, :, s % 128].cpu() for s in slots[lo:hi]])
                        torch.testing.assert_close(got, offset[..., 0], rtol=0, atol=0)
                    else:
                        anchors = key[torch.tensor([0, 128])].float()
                        a_scale = anchors.abs().amax(-1, keepdim=True) / 448
                        a_scale = torch.where(a_scale == 0, 1, a_scale)
                        a_code = (anchors / a_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                        decoded = a_code.float() * a_scale
                        offset = decoded[torch.arange(lo, hi) // 128]
                        if saved_anchor is not None:
                            torch.testing.assert_close(cache[4][2].view(torch.uint8).cpu(), saved_anchor)
                        saved_anchor = cache[4][2].view(torch.uint8).cpu().clone()
                        torch.testing.assert_close(cache[4][2].view(torch.uint8).cpu(), a_code[0].view(torch.uint8))
                        torch.testing.assert_close(cache[5][2].cpu(), a_scale[0, :, 0], rtol=1e-6, atol=0)
                    residual = kc - offset
                    expected_scale = residual.abs().amax(-1) / 448
                    expected_scale = torch.where(expected_scale == 0, 1, expected_scale)
                    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=2e-6, atol=1e-8)
                    decoded_k = codes.cpu().float() * scale.cpu()[..., None] + offset
                    self.assertTrue(torch.isfinite(decoded_k).all())
                    # Bound quantization loss by half a worst-case FP8 ULP.
                    max_error = (decoded_k - kc).abs().amax(-1)
                    self.assertTrue(torch.all(max_error <= expected_scale * 16.01))
                # Prefix/block transfer includes offset metadata, too.
                raw[pitch : 2 * pitch].copy_(raw[2 * pitch : 3 * pitch])
                for component in cache[4:]:
                    torch.testing.assert_close(component[1].view(torch.uint8), component[2].view(torch.uint8))
                preserved_prefix = raw[pitch : 2 * pitch].cpu().clone()
                # A different request opens the same physical page after recycle.
                raw[2 * pitch : 3 * pitch].zero_()
                replacement = torch.full((1, heads, dim), -7.0, dtype=torch.bfloat16, device="npu")
                self.quantize(replacement, torch.tensor([256], dtype=torch.int64, device="npu"), cache, mode)
                torch.npu.synchronize()
                if mode == "per_token_head_scalar":
                    torch.testing.assert_close(cache[4][2, :, 0].cpu(), torch.full((heads,), -7.0))
                else:
                    decoded_anchor = cache[4][2].cpu().float() * cache[5][2].cpu()[:, None]
                    torch.testing.assert_close(decoded_anchor, replacement.cpu().float()[0], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(raw[pitch : 2 * pitch].cpu(), preserved_prefix)
                torch.testing.assert_close(raw[:pitch].cpu(), before[:pitch])
                # Padding never initializes anchor metadata or reads a real page.
                saved = raw.cpu().clone()
                codes, scale = self.quantize(
                    torch.zeros((1, heads, dim), dtype=torch.bfloat16, device="npu"),
                    torch.tensor([-1], dtype=torch.int64, device="npu"),
                    cache,
                    mode,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(raw.cpu(), saved)
                self.assertTrue(torch.all(codes.cpu().view(torch.uint8) == 0))
                self.assertTrue(torch.all(scale.cpu() == 1))

    def test_graph_replay_refreshes_key_slots_and_anchor_lifetime(self):
        torch.manual_seed(189)
        heads, dim, pitch = 2, 128, 2 * 128 * 268 + 64
        spec = SimpleNamespace(num_kv_heads=heads, head_size=dim, head_size_v=dim)
        inputs = (torch.randn(5, heads, dim) * 2 + torch.arange(5)[:, None, None] * 3).bfloat16()
        for mode in ("per_token_head_scalar", "per_block_fp8_anchor"):
            with self.subTest(mode=mode):
                raw = torch.full((4 * pitch,), 37, dtype=torch.int8, device="npu")
                eager_raw = torch.empty_like(raw)
                cache = self.helpers.qfa_fp8_cache_views(raw, 4, spec, 128, page_stride_bytes=pitch, k_offset_mode=mode)
                eager_cache = self.helpers.qfa_fp8_cache_views(
                    eager_raw, 4, spec, 128, page_stride_bytes=pitch, k_offset_mode=mode
                )
                key = inputs[0:1].npu()
                slots = torch.tensor([256], dtype=torch.int64, device="npu")
                input_addresses = (key.data_ptr(), slots.data_ptr())
                for _ in range(3):
                    self.quantize(key, slots, cache, mode)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    codes, scale = self.quantize(key, slots, cache, mode)
                torch.npu.synchronize()
                output_addresses = (codes.data_ptr(), scale.data_ptr())
                raw.fill_(37)

                # One captured decode shape: open, append, padding, another
                # page, then the first page assigned to a different request.
                for step, slot in enumerate((256, 257, -1, 384, 256)):
                    if step == 4:
                        raw[2 * pitch : 3 * pitch].zero_()
                    key.copy_(inputs[step : step + 1].npu())
                    slots.copy_(torch.tensor([slot], dtype=torch.int64, device="npu"))
                    before = raw.cpu().clone()
                    eager_raw.copy_(raw)
                    graph.replay()
                    torch.npu.synchronize()
                    actual_codes, actual_scale = codes.cpu(), scale.cpu()
                    eager_codes, eager_scale = self.quantize(key, slots, eager_cache, mode)
                    torch.npu.synchronize()
                    torch.testing.assert_close(actual_codes.view(torch.uint8), eager_codes.cpu().view(torch.uint8))
                    torch.testing.assert_close(actual_scale, eager_scale.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(raw.cpu(), eager_raw.cpu())
                    self.assertEqual(input_addresses, (key.data_ptr(), slots.data_ptr()))
                    self.assertEqual(output_addresses, (codes.data_ptr(), scale.data_ptr()))
                    if slot < 0:
                        torch.testing.assert_close(raw.cpu(), before)
                        self.assertTrue(torch.all(actual_codes.view(torch.uint8) == 0))
                        self.assertTrue(torch.all(actual_scale == 1))
                        continue

                    block, position = divmod(slot, 128)
                    kc = inputs[step : step + 1].float()
                    if mode == "per_token_head_scalar":
                        offset = (kc.amax(-1, keepdim=True) + kc.amin(-1, keepdim=True)) * 0.5
                        torch.testing.assert_close(cache[4][block, :, position].cpu(), offset[0, :, 0], rtol=0, atol=0)
                        if position:
                            # Appending a scalar must not overwrite the first token.
                            first = inputs[0].float()
                            torch.testing.assert_close(
                                cache[4][block, :, 0].cpu(),
                                (first.amax(-1) + first.amin(-1)) * 0.5,
                                rtol=0,
                                atol=0,
                            )
                    else:
                        if position == 0:
                            anchor_scale = kc.abs().amax(-1, keepdim=True) / 448
                            anchor_scale = torch.where(anchor_scale == 0, 1, anchor_scale)
                            anchor_code = (kc / anchor_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                            torch.testing.assert_close(
                                cache[4][block].view(torch.uint8).cpu(), anchor_code[0].view(torch.uint8)
                            )
                            torch.testing.assert_close(cache[5][block].cpu(), anchor_scale[0, :, 0], rtol=1e-6, atol=0)
                        else:
                            torch.testing.assert_close(
                                raw[block * pitch : (block + 1) * pitch].cpu(),
                                before[block * pitch : (block + 1) * pitch],
                            )
                        offset = cache[4][block].cpu().float()[None] * cache[5][block].cpu()[None, :, None]
                    residual = kc - offset
                    expected_scale = residual.abs().amax(-1) / 448
                    expected_scale = torch.where(expected_scale == 0, 1, expected_scale)
                    torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-6, atol=1e-8)
                    reconstructed = actual_codes.float() * actual_scale[..., None] + offset
                    self.assertTrue(torch.all((reconstructed - kc).abs().amax(-1) <= expected_scale * 16.01))
                    # Cache writes follow the refreshed slot, preserving every
                    # other page, including the previous request's prefix page.
                    actual_raw = raw.cpu()
                    torch.testing.assert_close(actual_raw[: block * pitch], before[: block * pitch])
                    torch.testing.assert_close(actual_raw[(block + 1) * pitch :], before[(block + 1) * pitch :])


if __name__ == "__main__":
    unittest.main()
