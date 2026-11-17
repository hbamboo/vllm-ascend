# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone CPU contracts; run directly without initializing vLLM plugins."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def load_helpers(npu_ops):
    """Load unchanged production definitions without vLLM distributed setup.

    NPU integration tests supply real torch_npu and exercise the same helper,
    scatter kernel and QFA backend methods used by model inference.
    """
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    path = root / "vllm_ascend/attention/attention_c8_mxfp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
        )
    ]
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendC8MXFPAttentionBackendImpl")
    impl.bases = [ast.Name(id="object", ctx=ast.Load())]
    definitions.append(impl)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    namespace = {
        "torch": torch,
        "torch_npu": npu_ops,
        "math": math,
        "os": SimpleNamespace(environ={}),
        "notify_kv_cache_written": lambda: None,
        "record_attention_compute_start": lambda: None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def reference_mx_quant(x, *, dst_type, block_size=32):
    """CPU scale_alg=0 reference, including the even-padded scale byte."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], -1, block_size)
    amax = blocks.abs().amax(-1)
    exponent = (torch.floor(torch.log2(amax.clamp(min=2.0**-126))) - 8).clamp(-127, 127)
    scale = torch.exp2(exponent)
    quantized = (blocks / scale[..., None]).clamp(-448, 448).to(dst_type).reshape(shape)
    codes = (exponent + 127).to(torch.uint8)
    if codes.shape[-1] % 2:
        codes = torch.cat((codes, torch.zeros_like(codes[..., :1])), dim=-1)
    return quantized, codes.reshape(*shape[:-1], -1, 2)


class TestPerTokenKey(unittest.TestCase):
    def setUp(self):
        self.quantizer = MagicMock(side_effect=reference_mx_quant)
        self.helpers = load_helpers(SimpleNamespace(npu_dynamic_mx_quant=self.quantizer))

    def test_one_scale_covers_every_head_and_channel_of_each_token(self):
        for heads, dim in ((1, 128), (2, 128), (8, 128), (2, 256)):
            with self.subTest(heads=heads, dim=dim):
                # Disparate channel groups and token magnitudes distinguish
                # this from the old 32-channel per-token-group quantization.
                x = torch.ones(3, heads, dim, dtype=torch.bfloat16)
                x[:, :, :32] *= 0.125
                x *= torch.tensor([0.5, 8, 64]).reshape(3, 1, 1)
                quantized, scales = self.helpers.quantize_mxfp_key_per_token(x)
                expected, expected_scales = reference_mx_quant(
                    x.flatten(1), dst_type=torch.float8_e4m3fn, block_size=heads * dim
                )
                torch.testing.assert_close(quantized.view(torch.uint8), expected.reshape_as(x).view(torch.uint8))
                torch.testing.assert_close(scales.flatten(1), expected_scales[:, 0, :1].expand(3, heads * dim // 32))
                self.assertEqual(scales.shape, (3, heads, dim // 64, 2))
                self.assertTrue(scales.is_contiguous())
                self.assertEqual(self.quantizer.call_args.kwargs["block_size"], heads * dim)
                self.assertNotEqual(scales[0, 0, 0, 0], scales[1, 0, 0, 0])

    def test_noncontiguous_input(self):
        x = torch.randn(4, 2, 256, dtype=torch.bfloat16)[..., ::2]
        actual = self.helpers.quantize_mxfp_key_per_token(x)
        expected = self.helpers.quantize_mxfp_key_per_token(x.contiguous())
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8))

    def test_empty_input_does_not_call_quantizer(self):
        q, s = self.helpers.quantize_mxfp_key_per_token(torch.empty(0, 1, 128, dtype=torch.bfloat16))
        self.assertEqual(q.shape, (0, 1, 128))
        self.assertEqual(s.shape, (0, 1, 2, 2))
        self.quantizer.assert_not_called()

    def test_invalid_shapes_and_dtype(self):
        for shape in ((2, 128), (2, 1, 100), (2, 16, 128), (2, 0, 128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.helpers.quantize_mxfp_key_per_token(torch.empty(shape, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            self.helpers.quantize_mxfp_key_per_token(torch.zeros(2, 1, 128))

    def test_forward_changes_only_k_and_slices_padding(self):
        for mode in ("per_token", "per_token_group"):
            with self.subTest(mode=mode):
                quantizer = MagicMock(side_effect=reference_mx_quant)
                value_quantizer = MagicMock(side_effect=lambda x, *args: x.to(torch.float8_e4m3fn))
                helpers = load_helpers(SimpleNamespace(npu_dynamic_mx_quant=quantizer, npu_quantize=value_quantizer))
                impl = helpers.AscendC8MXFPAttentionBackendImpl()
                impl.k_quant_mode = mode
                impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
                impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 128
                impl.kv_sharing_target_layer_name = None
                impl.reshape_and_cache = MagicMock()
                output = torch.empty(3, 6, 128)
                impl._forward_mxfp8_attention = MagicMock(return_value=output)
                layer = SimpleNamespace(v_cache_scale_float_reciprocal=torch.ones(128, dtype=torch.bfloat16))
                query = torch.randn(4, 6, 128, dtype=torch.bfloat16)
                key = torch.randn(4, 1, 128, dtype=torch.bfloat16)
                value = torch.randn_like(key)
                metadata = SimpleNamespace(num_actual_tokens=3)
                cache = tuple(object() for _ in range(4))
                impl.forward(layer, query, key, value, cache, metadata, output)
                self.assertEqual(quantizer.call_count, 2)
                self.assertEqual(quantizer.call_args_list[0].args[0].shape, (3, 6, 128))
                self.assertNotIn("block_size", quantizer.call_args_list[0].kwargs)
                if mode == "per_token":
                    self.assertEqual(quantizer.call_args_list[1].kwargs["block_size"], 128)
                else:
                    self.assertNotIn("block_size", quantizer.call_args_list[1].kwargs)
                self.assertIs(value_quantizer.call_args.args[1], layer.v_cache_scale_float_reciprocal)
                self.assertEqual(impl.reshape_and_cache.call_args.args[0].shape, (3, 1, 128))


if __name__ == "__main__":
    unittest.main()
