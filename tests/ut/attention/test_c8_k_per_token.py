# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone CPU contracts; run directly without initializing vLLM plugins."""

import ast
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    parent = ast.parse((root / "vllm_ascend/attention/attention_v1.py").read_text(encoding="utf-8"))
    provider = next(n for n in parent.body if isinstance(n, ast.ClassDef) and n.name == "FIAV2ParamProvider")
    base = next(n for n in parent.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionBackendImpl")
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "_graph_metadata_layer_name")
    namespace["dataclass"] = dataclass
    module = ast.fix_missing_locations(ast.Module(body=[future, provider, method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    namespace["AscendC8MXFPAttentionBackendImpl"]._graph_metadata_layer_name = namespace["_graph_metadata_layer_name"]
    path = root / "vllm_ascend/attention/attention_c8_fia.py"
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        or isinstance(node, ast.ClassDef)
        and node.name == "AscendC8FIAPseudoAttentionBackendImpl"
    ]
    namespace.update(_EXTRA_CTX=SimpleNamespace(capturing=False), SWA_INT_MAX=2147483647)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
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


def reference_dynamic_quant(x, *, dst_type, quant_mode):
    if quant_mode != "pertoken":
        raise AssertionError("FIA KV must use per-token scales")
    scale = x.float().abs().amax(-1) / torch.finfo(dst_type).max
    return (x.float() / scale[..., None]).to(dst_type), scale


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
                quantized, scales = self.helpers.quantize_mxfp_per_token(x)
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
        actual = self.helpers.quantize_mxfp_per_token(x)
        expected = self.helpers.quantize_mxfp_per_token(x.contiguous())
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8))

    def test_empty_input_does_not_call_quantizer(self):
        q, s = self.helpers.quantize_mxfp_per_token(torch.empty(0, 1, 128, dtype=torch.bfloat16))
        self.assertEqual(q.shape, (0, 1, 128))
        self.assertEqual(s.shape, (0, 1, 2, 2))
        self.quantizer.assert_not_called()

    def test_invalid_shapes_and_dtype(self):
        for shape in ((2, 128), (2, 1, 100), (2, 16, 128), (2, 0, 128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.helpers.quantize_mxfp_per_token(torch.empty(shape, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            self.helpers.quantize_mxfp_per_token(torch.zeros(2, 1, 128))

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

    def test_forward_passes_unquantized_query_to_sink(self):
        impl = self.helpers.AscendC8MXFPAttentionBackendImpl()
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 128
        impl._forward_mxfp8_attention = MagicMock()
        query = torch.randn(4, 6, 256, dtype=torch.bfloat16)[..., ::2]
        sink = torch.randn(1, 128, dtype=torch.bfloat16)
        impl.forward(
            SimpleNamespace(sink_key=sink),
            query,
            None,
            None,
            tuple(object() for _ in range(4)),
            SimpleNamespace(num_actual_tokens=3),
            torch.empty(3, 6, 128),
        )
        passed = impl._forward_mxfp8_attention.call_args.kwargs
        self.assertIs(passed["sink_key"], sink)
        self.assertEqual(passed["sink_query"].data_ptr(), query.data_ptr())
        torch.testing.assert_close(passed["sink_query"], query[:3], rtol=0, atol=0)

    def test_qfa_sink_fuses_raw_query_and_writes_output_directly(self):
        impl = self.helpers.AscendC8MXFPAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 128
        impl.scale, impl.sliding_window = 128**-0.5, 4096
        query = torch.randn(3, 6, 128, dtype=torch.bfloat16)
        sink = torch.randn(1, 128, dtype=torch.bfloat16)
        attention = torch.ones_like(query)
        lse = torch.zeros(6, 3)
        qfa = MagicMock(return_value=(attention, lse))
        fused = MagicMock(side_effect=lambda *args: args[10][:3].fill_(2))
        with patch.dict(
            sys.modules,
            {
                "cann_ops_transformer.ops": SimpleNamespace(quant_flash_attn=qfa),
                "vllm_ascend.ops.triton.sink": SimpleNamespace(qfa_sink_apply_triton=fused),
            },
        ):
            for has_sink in (False, True):
                with self.subTest(has_sink=has_sink):
                    output = torch.full((4, 6, 128), -7, dtype=torch.bfloat16)
                    result = impl._run_qfa(
                        query.to(torch.float8_e4m3fn),
                        torch.full((3, 6, 2, 2), 127, dtype=torch.uint8),
                        (None, None, torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8)),
                        SimpleNamespace(block_tables=None),
                        cu_seqlens_q=torch.tensor([3], dtype=torch.int32),
                        seqused_kv=torch.tensor([3], dtype=torch.int32),
                        qfa_metadata=None,
                        max_seqlen_q=3,
                        mask_mode=0,
                        layout_q_descale="TND",
                        num_tokens=3,
                        output=output,
                        sink_query=query if has_sink else None,
                        sink_key=sink if has_sink else None,
                    )
                    self.assertIs(result, output)
                    self.assertEqual(qfa.call_args.kwargs["return_softmax_lse"], has_sink)
                    torch.testing.assert_close(output[:3], torch.full_like(query, 2 if has_sink else 1))
                    self.assertTrue(torch.all(output[3] == -7))
                    if has_sink:
                        self.assertIs(fused.call_args.args[0], query)
                        self.assertIsNone(fused.call_args.args[1])
                        self.assertIs(fused.call_args.args[3], lse)
                        self.assertIs(fused.call_args.args[4], sink)
                        self.assertIs(fused.call_args.args[10], output)
                    else:
                        fused.assert_not_called()

    def test_dynamic_v_scatter_and_decode_preserve_history(self):
        def scatter(cache, indices, payload):
            cache[indices[:, 0], indices[:, 1]] = payload

        ops = SimpleNamespace(
            npu_dynamic_mx_quant=self.quantizer,
            npu_dynamic_quant=MagicMock(side_effect=reference_dynamic_quant),
            npu_scatter_nd_update_=scatter,
            npu_fused_infer_attention_score_v2=MagicMock(side_effect=lambda **kw: (kw["query"].clone(), None)),
        )
        helpers = load_helpers(ops)
        impl = helpers.AscendC8FIAPseudoAttentionBackendImpl()
        impl.kv_sharing_target_layer_name = None
        impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = 6, 1, 128, 128**-0.5
        spec = SimpleNamespace(num_kv_heads=1, head_size=128, head_size_v=128)
        pitch = helpers.mxfp_packet_size_bytes(1, 512, 128, 128)
        cache = helpers.fia_pseudo_cache_views(
            torch.zeros(3 * pitch, dtype=torch.int8), 3, spec, 512, page_stride_bytes=pitch
        )
        layer = SimpleNamespace(v_cache_scale=torch.full((128,), 127, dtype=torch.uint8))
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[3],
            slot_mapping=torch.tensor([1023, 1024, 514]),
            causal=True,
            block_tables=None,
            seq_lens_list=[3],
            attn_mask=None,
        )
        k = torch.ones(3, 1, 128, dtype=torch.bfloat16)
        v = k * torch.tensor([0.5, 8, 64], dtype=torch.bfloat16).reshape(3, 1, 1)
        q = torch.zeros(3, 6, 128, dtype=torch.bfloat16)
        impl.forward(layer, q, k, v, cache, metadata, output=torch.empty_like(q))
        _, expected = reference_dynamic_quant(v.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        slots = metadata.slot_mapping
        torch.testing.assert_close(cache[3][slots // 512, slots % 512], expected)
        self.assertEqual(ops.npu_dynamic_quant.call_count, 2)
        self.assertEqual(ops.npu_dynamic_quant.call_args.kwargs["quant_mode"], "pertoken")
        self.quantizer.assert_not_called()
        history = cache[3].clone()
        metadata.actual_seq_lengths_q = [1]
        metadata.slot_mapping = torch.tensor([515])
        impl.forward(layer, q[:1], k[:1], v[:1] * 4, cache, metadata, output=torch.empty_like(q[:1]))
        torch.testing.assert_close(cache[3][slots // 512, slots % 512], history[slots // 512, slots % 512])
        self.assertEqual(ops.npu_fused_infer_attention_score_v2.call_args.kwargs["value_quant_mode"], 4)
        self.assertIs(ops.npu_fused_infer_attention_score_v2.call_args.kwargs["dequant_scale_value"], cache[3])
        # Verify the model graph task resolves current sequence lengths via
        # the real base method/provider rather than freezing Python lists.
        context = impl.forward.__globals__
        context["_EXTRA_CTX"].capturing = True
        context["_FIA_V2_WORKSPACE_KEY"] = "fia"
        context["get_capture_resource"] = lambda *args: None
        context["register_task"] = MagicMock()
        impl._use_max_workspace_for_fia_graph = False
        layer.layer_name = "attention"
        impl.forward(layer, q[:1], None, None, cache, metadata, output=torch.empty_like(q[:1]))
        provider = context["register_task"].call_args.args[2]
        metadata.seq_lens_list = [9]
        self.assertEqual(provider.resolve({"attention": metadata})["actual_seq_kvlen"], [9])


if __name__ == "__main__":
    unittest.main()
