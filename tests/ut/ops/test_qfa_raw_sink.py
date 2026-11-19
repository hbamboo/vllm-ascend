# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU launch-contract tests for raw-Q and quantized-Q QFA sink correction."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


class TestQFARawSinkLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        spec = importlib.util.spec_from_file_location("qfa_raw_sink_test", root / "vllm_ascend/ops/triton/sink.py")
        cls.sink = importlib.util.module_from_spec(spec)
        triton = SimpleNamespace(jit=lambda fn: fn, cdiv=lambda x, y: (x + y - 1) // y)
        with patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=SimpleNamespace(constexpr=int)),
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=MagicMock()),
            },
        ):
            spec.loader.exec_module(cls.sink)

    def make_case(self, tokens=12, dim_stride=1):
        heads, kv_heads, dim = 6, 1, 128
        query = torch.empty(tokens, heads + 1, dim * dim_stride, dtype=torch.bfloat16)[:, :heads, ::dim_stride]
        attn = torch.empty(tokens, heads, dim, dtype=torch.bfloat16)
        lse = torch.empty(heads, tokens + 3)[:, :tokens]
        sink_key = torch.empty(kv_heads, dim)
        output = torch.empty_like(attn)
        return SimpleNamespace(**locals())

    def launch(self, case, query_scale, kernel):
        with patch.object(self.sink, "_triton_qfa_sink_apply", kernel):
            self.sink.qfa_sink_apply_triton(
                case.query,
                query_scale,
                case.attn,
                case.lse,
                case.sink_key,
                case.dim**-0.5,
                case.tokens,
                case.heads,
                case.kv_heads,
                case.dim,
                case.output,
            )

    def test_raw_query_preserves_strides_and_direct_output(self):
        case = self.make_case()
        kernel = MagicMock()
        self.launch(case, None, kernel)
        kernel.__getitem__.assert_called_once_with((3,))
        args, kwargs = kernel.__getitem__.return_value.call_args
        self.assertEqual(args[0].data_ptr(), case.query.data_ptr())
        self.assertIs(args[1], args[0])  # Unused pointer for the constexpr raw-Q branch.
        self.assertIs(args[2], case.attn)
        self.assertEqual(args[3].data_ptr(), case.lse.data_ptr())
        self.assertIs(args[4], case.sink_key)
        self.assertIs(args[6], case.output)
        self.assertFalse(kwargs["HAS_Q_SCALE"])
        self.assertEqual(kwargs["Q_SCALE_TOKEN_STRIDE"], 0)
        self.assertEqual(kwargs["Q_SCALE_HEAD_STRIDE"], 0)
        self.assertEqual(kwargs["Q_TOKEN_STRIDE"], case.query.stride(0))
        self.assertEqual(kwargs["Q_HEAD_STRIDE"], case.query.stride(1))
        self.assertEqual(kwargs["LSE_HEAD_STRIDE"], case.lse.stride(0))
        self.assertEqual(kwargs["LSE_TOKEN_STRIDE"], case.lse.stride(1))
        self.assertTrue(kwargs["HAS_SINK"])
        self.assertFalse(kwargs["HAS_V_SCALE"])

    def test_raw_query_materializes_nonunit_dim_stride(self):
        case = self.make_case(dim_stride=2)
        case.query.copy_(torch.randn_like(case.query))
        kernel = MagicMock()
        self.launch(case, None, kernel)
        args, kwargs = kernel.__getitem__.return_value.call_args
        self.assertNotEqual(args[0].data_ptr(), case.query.data_ptr())
        self.assertTrue(args[0].is_contiguous())
        torch.testing.assert_close(args[0], case.query, rtol=0, atol=0)
        self.assertEqual(kwargs["Q_TOKEN_STRIDE"], case.heads * case.dim)
        self.assertEqual(kwargs["Q_HEAD_STRIDE"], case.dim)
        self.assertFalse(kwargs["HAS_Q_SCALE"])

    def test_quantized_query_retains_token_and_head_scale_strides(self):
        case = self.make_case()
        for shape in ((case.tokens,), (case.tokens, case.heads)):
            with self.subTest(shape=shape):
                query_scale = torch.empty(shape)
                kernel = MagicMock()
                self.launch(case, query_scale, kernel)
                args, kwargs = kernel.__getitem__.return_value.call_args
                self.assertIs(args[1], query_scale)
                self.assertTrue(kwargs["HAS_Q_SCALE"])
                self.assertEqual(kwargs["Q_SCALE_TOKEN_STRIDE"], query_scale.stride(0))
                self.assertEqual(kwargs["Q_SCALE_HEAD_STRIDE"], query_scale.stride(1) if query_scale.ndim == 2 else 0)

    def test_empty_query_does_not_initialize_or_launch(self):
        kernel = MagicMock()
        with patch.object(self.sink, "init_device_properties_triton") as init:
            self.launch(self.make_case(tokens=0), None, kernel)
        kernel.__getitem__.assert_not_called()
        init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
