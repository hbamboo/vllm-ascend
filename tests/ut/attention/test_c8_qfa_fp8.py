# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for QFA FP8 pages; run directly without loading vLLM."""

import ast
import math
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


def load_cache_helpers():
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    path = root / "vllm_ascend/attention/attention_c8_mxfp.py"
    names = {"qfa_fp8_cache_views", "mxfp_cache_views_for_spec"}
    definitions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class TestQfaFp8CacheViews(unittest.TestCase):
    def setUp(self):
        self.helpers = load_cache_helpers()
        self.spec = SimpleNamespace(num_kv_heads=2, head_size=128, head_size_v=128, block_size=512)
        self.kernel_block_size = 128
        self.kv_bytes = self.spec.num_kv_heads * self.kernel_block_size * self.spec.head_size
        self.scale_bytes = self.spec.num_kv_heads * self.kernel_block_size * 4
        self.payload_bytes = 2 * self.kv_bytes + self.scale_bytes + self.spec.num_kv_heads * 4

    def test_scheduler_pages_split_into_four_padded_kernel_pages(self):
        pitch = self.payload_bytes + 64
        self.spec.page_size_bytes = 4 * pitch
        raw = torch.zeros(2 * self.spec.page_size_bytes, dtype=torch.int8)
        key, value, scale, v_scale = self.helpers.mxfp_cache_views_for_spec(
            raw, self.spec, self.kernel_block_size, qfa_fp8=True
        )

        self.assertEqual(key.shape, (8, 2, 128, 128))
        self.assertEqual(value.shape, key.shape)
        self.assertEqual(scale.shape, (8, 2, 128))
        self.assertEqual(v_scale.shape, (8, 2))
        self.assertEqual((key.dtype, value.dtype, scale.dtype), (torch.float8_e4m3fn,) * 2 + (torch.float32,))
        self.assertEqual(v_scale.dtype, torch.float32)
        for view in (key, value, scale, v_scale):
            self.assertEqual(view.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())
            self.assertEqual(view.stride(0) * view.element_size(), pitch)
            self.assertEqual(view[4].data_ptr() - view[0].data_ptr(), self.spec.page_size_bytes)

        # Every section remains separate, including when the physical page has padding.
        key.view(torch.uint8).fill_(17)
        value.view(torch.uint8).fill_(29)
        scale.fill_(2.5)
        v_scale.fill_(0.75)
        self.assertTrue(torch.all(key.view(torch.uint8) == 17))
        self.assertTrue(torch.all(value.view(torch.uint8) == 29))
        self.assertTrue(torch.all(scale == 2.5))
        self.assertTrue(torch.all(v_scale == 0.75))
        self.assertTrue(torch.all(raw.view(8, pitch)[:, self.payload_bytes :] == 0))

        # Recycling scheduler page 0 clears its four kernel pages, not page 1.
        neighbor = raw[self.spec.page_size_bytes :].clone()
        raw[: self.spec.page_size_bytes].zero_()
        for view in (key, value):
            self.assertTrue(torch.all(view[:4].view(torch.uint8) == 0))
        self.assertTrue(torch.all(scale[:4] == 0))
        self.assertTrue(torch.all(v_scale[:4] == 0))
        torch.testing.assert_close(raw[self.spec.page_size_bytes :], neighbor)

    def test_aligned_slice_preserves_absolute_storage_offsets(self):
        prefix_bytes = 16
        pitch = self.payload_bytes + 64
        allocation = torch.full((prefix_bytes + 2 * pitch + 16,), 73, dtype=torch.int8)
        raw = allocation[prefix_bytes : prefix_bytes + 2 * pitch]
        key, value, scale, v_scale = self.helpers.qfa_fp8_cache_views(
            raw, 2, self.spec, self.kernel_block_size, page_stride_bytes=pitch
        )

        self.assertEqual(key.storage_offset(), prefix_bytes)
        self.assertEqual(value.storage_offset(), prefix_bytes + self.kv_bytes)
        self.assertEqual(scale.storage_offset() * scale.element_size(), prefix_bytes + 2 * self.kv_bytes)
        self.assertEqual(
            v_scale.storage_offset() * v_scale.element_size(), prefix_bytes + 2 * self.kv_bytes + self.scale_bytes
        )
        key.view(torch.uint8).zero_()
        value.view(torch.uint8).zero_()
        scale.zero_()
        v_scale.zero_()
        self.assertTrue(torch.all(allocation[:prefix_bytes] == 73))
        self.assertTrue(torch.all(allocation[prefix_bytes + 2 * pitch :] == 73))

        with self.assertRaisesRegex(ValueError, "aligned complete pages"):
            self.helpers.qfa_fp8_cache_views(
                allocation[1 : 1 + 2 * pitch], 2, self.spec, self.kernel_block_size, page_stride_bytes=pitch
            )

    def test_rejects_unsupported_head_dim_and_incomplete_page(self):
        raw = torch.zeros(self.payload_bytes, dtype=torch.int8)
        spec = SimpleNamespace(num_kv_heads=2, head_size=64, head_size_v=64)
        with self.assertRaisesRegex(ValueError, "128-wide K/V heads"):
            self.helpers.qfa_fp8_cache_views(raw, 1, spec, self.kernel_block_size)
        with self.assertRaisesRegex(ValueError, "aligned complete pages"):
            self.helpers.qfa_fp8_cache_views(raw[:-4], 1, self.spec, self.kernel_block_size)


class TestQfaSlidingPlans(unittest.TestCase):
    def test_full_and_sliding_plans_keep_distinct_windows_in_decode_and_prefill(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        load_helpers = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
        helpers = load_helpers(SimpleNamespace())
        impl = helpers.AscendC8MXFPAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 128
        impl.scale = 128**-0.5
        for q_len in (1, 6, 129):
            with self.subTest(q_len=q_len):
                query = torch.zeros(q_len, 6, 128, dtype=torch.float8_e4m3fn)
                query_scale = torch.full((q_len, 6, 2, 2), 127, dtype=torch.uint8)
                cache = (query, query, torch.ones(1, dtype=torch.uint8), torch.ones(1, dtype=torch.uint8))
                metadata = SimpleNamespace(
                    causal=True,
                    num_actual_tokens=q_len,
                    max_query_len=q_len,
                    query_start_loc_gpu=torch.tensor([0, q_len], dtype=torch.int32),
                    seq_lens_gpu=torch.tensor([1025], dtype=torch.int32),
                    block_tables=torch.zeros(1, 9, dtype=torch.int32),
                    attn_mask=torch.ones(2048, 2048, dtype=torch.int8),
                )
                native = SimpleNamespace(
                    quant_flash_attn_metadata=MagicMock(return_value=torch.ones(1)),
                    quant_flash_attn=MagicMock(return_value=(torch.zeros_like(query), None)),
                )
                with patch.dict(sys.modules, {"cann_ops_transformer.ops": native}):
                    for window in (None, 4, 7, 4):
                        impl.sliding_window = window
                        impl._forward_mxfp8_attention(
                            query, query_scale, cache, metadata, torch.empty(q_len, 6, 128, dtype=torch.bfloat16)
                        )
                        call = native.quant_flash_attn.call_args.kwargs
                        expected_mask = 4 if window else (0 if q_len == 1 else 3)
                        self.assertEqual(call["mask_mode"], expected_mask)
                        self.assertEqual(call["win_left"], window - 1 if window else -1)
                        self.assertEqual(call["win_right"], 0 if window else -1)
                        self.assertEqual(call["layout_q_descale"], "N2TGD" if q_len == 1 else "TND")
                self.assertEqual(native.quant_flash_attn_metadata.call_count, 3)
                for call, window in zip(native.quant_flash_attn_metadata.call_args_list, (None, 4, 7)):
                    self.assertEqual(call.args[3], 1)
                    self.assertEqual(call.kwargs["win_left"], window - 1 if window else -1)
                    self.assertEqual(call.kwargs["win_right"], 0 if window else -1)


if __name__ == "__main__":
    unittest.main()
