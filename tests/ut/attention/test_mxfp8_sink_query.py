# SPDX-License-Identifier: Apache-2.0
"""CPU numerical checks for the query used by C8 learned-key sinks.

Load the production functions without initializing the vLLM/NPU backend.
The quantizer is mocked; FP8 conversion and all sink arithmetic use torch.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class TestMXFP8SinkQuery(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_c8_mxfp.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        constants = {
            "MXFP8_GROUP_SIZE",
            "FP32_MANTISSA_BITS",
            "E8M0_MIN_FP32_BITS",
            "E8M0_NAN_CODE",
            "FP32_QNAN_BITS",
        }
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
            )
            or isinstance(node, ast.FunctionDef)
            and node.name == "_dequantize_mxfp8_query"
        ]
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AscendC8MXFPAttentionBackendImpl"
        )
        nodes.append(
            next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        )
        future_annotations = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future_annotations, *nodes], type_ignores=[]))
        self.quantize = Mock()
        self.namespace = {
            "os": os,
            "torch": torch,
            "torch_npu": SimpleNamespace(npu_dynamic_mx_quant=self.quantize),
            "record_attention_compute_start": Mock(),
        }
        exec(compile(module, str(source), "exec"), self.namespace)

    def test_dequantization_matches_native_e8m0_for_all_scale_codes(self):
        scale_bytes = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(1, 1, 128, 2)
        query = torch.ones(1, 1, 8192).to(torch.float8_e4m3fn)
        expected = scale_bytes.view(torch.float8_e8m0fnu).float().reshape(-1).repeat_interleave(32).reshape(query.shape)
        for scale in (scale_bytes, scale_bytes.view(torch.float8_e8m0fnu)):
            with self.subTest(dtype=scale.dtype):
                actual = self.namespace["_dequantize_mxfp8_query"](query, scale)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_empty_query_dequantization_preserves_shape(self):
        query = torch.empty(0, 6, 128).to(torch.float8_e4m3fn)
        scales = torch.empty(0, 6, 2, 2, dtype=torch.uint8)
        actual = self.namespace["_dequantize_mxfp8_query"](query, scales)
        self.assertEqual(actual.shape, query.shape)
        self.assertEqual(actual.dtype, torch.float32)

    def _forward(self, sink_key, *, disable_sink=False, byte_scales=False):
        num_tokens, num_heads, num_kv_heads, head_dim = 2, 6, 2, 128
        query = torch.full((num_tokens + 3, num_heads, head_dim), 19.0, dtype=torch.bfloat16)
        codes = torch.arange(num_tokens * num_heads * head_dim).remainder(15).sub(7).float() / 2
        quant_query = codes.reshape(num_tokens, num_heads, head_dim).to(torch.float8_e4m3fn)
        scale_bytes = (
            torch.arange(num_tokens * num_heads * 4)
            .remainder(7)
            .add(124)
            .to(torch.uint8)
            .reshape(num_tokens, num_heads, 2, 2)
        )
        scales = scale_bytes if byte_scales else scale_bytes.view(torch.float8_e8m0fnu)
        self.quantize.return_value = quant_query, scales
        output = torch.empty(num_tokens, num_heads, head_dim)
        impl = SimpleNamespace(
            vllm_config=SimpleNamespace(kv_transfer_config=None),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            scale=0.125,
            _forward_mxfp8_attention=Mock(return_value=output),
        )
        metadata = SimpleNamespace(num_actual_tokens=num_tokens)
        with patch.dict(os.environ, {"C8_DISABLE_SINK": "1" if disable_sink else "0"}):
            result = self.namespace["forward"](
                impl, SimpleNamespace(sink_key=sink_key), query, None, None, (None,) * 4, metadata, output
            )
        self.assertIs(result, output)
        forwarded = impl._forward_mxfp8_attention.call_args
        self.assertIs(forwarded.args[0], quant_query)
        self.assertIs(forwarded.args[1], scales)
        torch.testing.assert_close(self.quantize.call_args.args[0], query[:num_tokens])
        return forwarded.kwargs["sink_logits"], quant_query, scale_bytes, impl

    def test_sink_uses_scaled_fp8_query_with_gqa_and_padding(self):
        sink_key = torch.linspace(-1, 2, 256).reshape(2, 128)
        for byte_scales in (False, True):
            with self.subTest(byte_scales=byte_scales):
                actual, quant_query, scale_bytes, impl = self._forward(sink_key, byte_scales=byte_scales)
                # Independent reference: native E8M0 conversion, per-element
                # scale expansion, then a head-wise dot product.
                scales = scale_bytes.view(torch.float8_e8m0fnu).float().flatten(-2).repeat_interleave(32, dim=-1)
                dequant_query = quant_query.float() * scales
                sink_per_head = sink_key.repeat_interleave(3, dim=0)
                expected = torch.einsum("thd,hd->th", dequant_query, sink_per_head) * impl.scale
                torch.testing.assert_close(actual, expected)
                raw_query_result = 19.0 * sink_per_head.sum(-1) * impl.scale
                self.assertFalse(torch.allclose(actual, raw_query_result.expand_as(actual)))
                unscaled_result = torch.einsum("thd,hd->th", quant_query.float(), sink_per_head) * impl.scale
                self.assertFalse(torch.allclose(actual, unscaled_result))

    def test_missing_or_disabled_sink_skips_dequantization(self):
        for sink_key, disabled in ((None, False), (torch.ones(2, 128), True)):
            with self.subTest(disabled=disabled):
                with patch.dict(self.namespace, {"_dequantize_mxfp8_query": Mock(side_effect=AssertionError)}):
                    sink_logits, _, _, _ = self._forward(sink_key, disable_sink=disabled)
                self.assertIsNone(sink_logits)


if __name__ == "__main__":
    unittest.main()
