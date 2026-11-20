# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone CPU contracts for offset pages and score compensation."""

import ast
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class TestQfaKOffsetPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = runpy.run_path(str(Path(__file__).with_name("test_c8_qfa_fp8.py")))["load_cache_helpers"]()
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        path = root / "vllm_ascend/worker/utils.py"
        copy = next(
            node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "copy_kv_cache_blocks_inplace"
        )
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, copy], type_ignores=[]))
        namespace = {"torch": torch}
        exec(compile(module, str(path), "exec"), namespace)
        cls.copy_pages = staticmethod(namespace[copy.name])

    def test_sections_survive_page_copy_and_recycling(self):
        for mode in ("per_token_head_scalar", "per_block_fp8_anchor"):
            for heads in (1, 2, 7):
                with self.subTest(mode=mode, heads=heads):
                    # 268 bytes/token/head reserves either offset mode, even
                    # before hybrid alignment supplies additional page padding.
                    pitch = heads * 128 * 268
                    spec = SimpleNamespace(
                        num_kv_heads=heads, head_size=128, head_size_v=128,
                        block_size=512, page_size_bytes=pitch * 4,
                    )
                    guard = 32
                    backing = torch.full((2 * spec.page_size_bytes + 2 * guard,), 29, dtype=torch.int8)
                    raw = backing[guard:-guard]
                    cache = self.helpers.mxfp_cache_views_for_spec(
                        raw, spec, 128, qfa_fp8=True, k_offset_mode=mode
                    )
                    self.assertEqual(len(cache), 5 if mode == "per_token_head_scalar" else 6)
                    ranges = []
                    for index, view in enumerate(cache):
                        start = view[0].data_ptr() - raw.data_ptr()
                        end = start + view[0].numel() * view.element_size()
                        self.assertLessEqual(end, pitch)
                        self.assertEqual(view.stride(0) * view.element_size(), pitch)
                        self.assertTrue(all(end <= a or start >= b for a, b in ranges))
                        ranges.append((start, end))
                        view[:4].view(torch.uint8).fill_(index + 1)
                    # Prefix/block copy transfers offsets with the original KV
                    # packet, and scheduler-page zeroing covers all four kernels.
                    self.copy_pages([cache], 2, [SimpleNamespace(src_block_id=0, dst_block_id=1)])
                    for view in cache:
                        torch.testing.assert_close(view[:4].view(torch.uint8), view[4:].view(torch.uint8))
                    saved = raw[spec.page_size_bytes:].clone()
                    raw[:spec.page_size_bytes].zero_()
                    for view in cache:
                        self.assertTrue(torch.all(view[:4].view(torch.uint8) == 0))
                    torch.testing.assert_close(raw[spec.page_size_bytes:], saved)
                    self.assertTrue(torch.all(backing[:guard] == 29))
                    self.assertTrue(torch.all(backing[-guard:] == 29))

    def test_legacy_page_does_not_silently_borrow_v_scale_or_padding(self):
        spec = SimpleNamespace(num_kv_heads=1, head_size=128, head_size_v=128)
        legacy_pitch = 128 * 264
        raw = torch.zeros(legacy_pitch, dtype=torch.int8)
        legacy = self.helpers.qfa_fp8_cache_views(raw, 1, spec, 128, page_stride_bytes=legacy_pitch)
        self.assertEqual(len(legacy), 4)
        self.assertEqual(legacy[3].storage_offset() * 4, 2 * 128 * 128 + 128 * 4)
        with self.assertRaisesRegex(ValueError, "aligned complete pages"):
            self.helpers.qfa_fp8_cache_views(
                raw, 1, spec, 128, page_stride_bytes=legacy_pitch, k_offset_mode="per_token_head_scalar"
            )


class TestQfaKOffsetScores(unittest.TestCase):
    def test_compensation_restores_logits_and_keeps_sink_coordinates(self):
        torch.manual_seed(21)
        q = torch.randn(5, 128, dtype=torch.float64)
        residual = torch.randn(257, 128, dtype=torch.float64)
        sink = torch.randn(128, dtype=torch.float64)
        for scalar in (True, False):
            if scalar:
                offset = torch.randn(257, 1, dtype=torch.float64)
                correction = q.sum(-1, keepdim=True) * offset.t()
            else:
                anchors = torch.randn(3, 128, dtype=torch.float64)
                offset = anchors[torch.arange(257) // 128]
                correction = q @ offset.t()
            expected = q @ (residual + offset).t() / 128**0.5
            actual = (q @ residual.t() + correction) / 128**0.5
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            sink_score = q @ sink / 128**0.5
            actual = torch.cat((actual, sink_score[:, None]), dim=-1)
            expected = torch.cat((expected, sink_score[:, None]), dim=-1)
            torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1), atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
