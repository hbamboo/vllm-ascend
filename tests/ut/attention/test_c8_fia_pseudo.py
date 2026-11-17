# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for dynamic per-token FP8 KV with paged FIA scales.

Device calls are mocked; FP8 encoding, page geometry, and attention arithmetic
use real CPU torch. Run directly to avoid initializing vLLM or torch_npu.
"""

import ast
import math
import unittest
from dataclasses import make_dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_nodes(path, namespace, kinds, names=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    nodes = [n for n in tree.body if isinstance(n, kinds) and (names is None or getattr(n, "name", None) in names)]
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def reference_quantize(x, *, dst_type, quant_mode):
    if x.ndim != 2 or quant_mode != "pertoken" or dst_type != torch.float8_e4m3fn:
        raise AssertionError("FIA mode 4 needs one scale per flattened token")
    scale = x.float().abs().amax(dim=-1) / torch.finfo(dst_type).max
    scale = torch.where(scale == 0, 1.0, scale)
    return (x.float() / scale[:, None]).clamp(-448, 448).to(dst_type), scale


class TestC8FIAPseudo(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.root = Path(__file__).resolve().parents[3]
        base = type("Base", (), {"get_graph_metadata_layer_name": lambda impl, name: name})
        self.fia = Mock(side_effect=self.reference_fia)
        self.fia.out = Mock()
        self.scatter = Mock(side_effect=self.scatter_nd)
        self.dynamic_quant = Mock(side_effect=reference_quantize)
        self.context = SimpleNamespace(capturing=False)
        self.register = Mock()
        self.sink = Mock()
        self.namespace = {
            "torch": torch,
            "math": math,
            "torch_npu": SimpleNamespace(
                npu_dynamic_quant=self.dynamic_quant,
                npu_quantize=Mock(side_effect=AssertionError("FIA must not use static V scales")),
                npu_scatter_nd_update_=self.scatter,
                npu_fused_infer_attention_score_v2=self.fia,
                npu_dynamic_mx_quant=Mock(side_effect=AssertionError("Q must stay BF16")),
                _npu_fused_infer_attention_score_v2_get_max_workspace=Mock(),
            ),
            "AscendC8MXFPAttentionBackend": base,
            "AscendC8MXFPAttentionBackendImpl": base,
            "_EXTRA_CTX": self.context,
            "is_950": lambda: True,
            "record_attention_compute_start": Mock(),
            "notify_kv_cache_written": Mock(),
            "sink_apply_triton": self.sink,
            "register_task": self.register,
            "get_capture_resource": lambda *args: "workspace",
            "_FIA_V2_WORKSPACE_KEY": "fia",
            "SWA_INT_MAX": 2147483647,
            "FIAV2ParamProvider": lambda name: SimpleNamespace(layer_name=name),
        }
        load_nodes(
            self.root / "vllm_ascend/attention/attention_c8_mxfp.py", self.namespace, (ast.Assign, ast.FunctionDef)
        )
        load_nodes(
            self.root / "vllm_ascend/attention/attention_c8_fia.py",
            self.namespace,
            (ast.Assign, ast.FunctionDef, ast.ClassDef),
        )

    @staticmethod
    def scatter_nd(cache, indices, values):
        cache[indices[:, 0], indices[:, 1]] = values
        return cache

    def make_cache(self, *, padded=False, ratio=1, heads=2, dim=128, pages=4, offset=32):
        spec = SimpleNamespace(num_kv_heads=heads, head_size=dim, head_size_v=dim, block_size=512 * ratio)
        payload = self.namespace["mxfp_packet_size_bytes"](heads, 512, dim, dim)
        spec.page_size_bytes = ratio * (payload + (64 if padded else 0))
        backing = torch.zeros(offset + pages * spec.page_size_bytes + 32, dtype=torch.int8)
        raw = backing[offset:-32]
        cache = self.namespace["mxfp_cache_views_for_spec"](raw, spec, 512, fia_pseudo=True)
        return cache, raw, spec

    @staticmethod
    def reference_fia(**kwargs):
        q = kwargs["query"].float()
        head_dim, q_heads = q.shape[-1], q.shape[-2]
        kv_heads = kwargs["num_key_value_heads"]
        block_size = kwargs["block_size"]
        outputs, lses = [], []
        start = 0
        for batch, (end, count) in enumerate(zip(kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"])):
            positions = torch.arange(count)
            pages = kwargs["block_table"][batch, positions // block_size].long()
            offsets = positions % block_size
            k = kwargs["key"].float()[pages, offsets].view(count, kv_heads, head_dim)
            v = kwargs["value"].float()[pages, offsets].view(count, kv_heads, head_dim)
            k *= kwargs["dequant_scale_key"][pages, offsets, None, None]
            v *= kwargs["dequant_scale_value"][pages, offsets, None, None]
            k, v = k.to(torch.bfloat16).float(), v.to(torch.bfloat16).float()
            k, v = (x.repeat_interleave(q_heads // kv_heads, dim=1) for x in (k, v))
            scores = torch.einsum("thd,shd->hts", q[start:end], k) * kwargs["softmax_scale"]
            if kwargs["sparse_mode"] == 3:
                mask = positions[None] > torch.arange(end - start)[:, None] + count - (end - start)
                scores.masked_fill_(mask[None], float("-inf"))
            outputs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), v).to(torch.bfloat16))
            lses.append(scores.logsumexp(-1).t().unsqueeze(-1))
            start = end
        return torch.cat(outputs), torch.cat(lses)

    def setup_forward(self):
        cache, _, _ = self.make_cache()
        impl = self.namespace["AscendC8FIAPseudoAttentionBackendImpl"]()
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 2, 128
        impl.scale, impl.sliding_window = 128**-0.5, None
        impl.kv_sharing_target_layer_name = None
        impl._use_max_workspace_for_fia_graph = False
        layer = SimpleNamespace(layer_name="layer")
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[2],
            seq_lens_list=[2],
            block_tables=torch.tensor([[1]], dtype=torch.int32),
            slot_mapping=torch.tensor([512, 513], dtype=torch.int32),
            causal=True,
            attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1),
        )
        q = torch.randn(2, 6, 128, dtype=torch.bfloat16)
        k, v = (torch.randn(2, 2, 128, dtype=torch.bfloat16) for _ in range(2))
        return impl, layer, metadata, cache, q, k, v, torch.empty_like(q)

    def test_all_components_stay_within_their_physical_page(self):
        for padded, ratio, heads in ((False, 1, 1), (True, 1, 2), (True, 2, 2)):
            with self.subTest(padded=padded, ratio=ratio, heads=heads):
                cache, raw, spec = self.make_cache(padded=padded, ratio=ratio, heads=heads)
                pitch = spec.page_size_bytes // ratio
                self.assertEqual(cache[0].shape, (4 * ratio, 512, heads * 128))
                self.assertEqual(cache[2].shape, (4 * ratio, 512))
                self.assertEqual(cache[2].dtype, torch.float32)
                self.assertEqual(cache[3].dtype, torch.float32)
                for page in range(4 * ratio):
                    begin, end = raw.data_ptr() + page * pitch, raw.data_ptr() + (page + 1) * pitch
                    ranges = []
                    for tensor in cache:
                        self.assertEqual(tensor.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())
                        self.assertTrue(tensor[page].is_contiguous())
                        self.assertEqual(tensor.stride(0) * tensor.element_size(), pitch)
                        lo = tensor.data_ptr() + page * pitch
                        hi = lo + tensor[page].numel() * tensor.element_size()
                        self.assertTrue(begin <= lo < hi <= end)
                        ranges.append((lo, hi))
                    ranges.sort()
                    self.assertTrue(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])))

    def test_factory_rejects_too_small_pages_and_misaligned_float_scales(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            self.make_cache(heads=1, dim=64)
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.make_cache(offset=33)

    def test_default_factory_retains_qfa_nz_packet_layout(self):
        _, raw, spec = self.make_cache()
        cache = self.namespace["mxfp_cache_views_for_spec"](raw, spec, 512)
        self.assertEqual(cache[0].shape, (4, 2, 4, 512, 32))
        self.assertTrue(all(t.stride(0) == spec.page_size_bytes for t in cache))
        self.assertEqual(cache[3].dtype, torch.uint8)

    def test_forward_restores_tnd_and_writes_dynamic_scales(self):
        impl, layer, meta, cache, q, k, v, out = self.setup_forward()
        k[0, 0, 0] = 1000
        impl.reshape_and_cache = Mock(wraps=impl.reshape_and_cache)
        result = impl.forward(layer, q, k, v, cache, meta, out)
        kwargs = self.fia.call_args.kwargs
        self.assertIs(result, out)
        torch.testing.assert_close(kwargs["query"], q, rtol=0, atol=0)
        self.assertEqual(kwargs["query"].dtype, torch.bfloat16)
        self.assertIs(kwargs["dequant_scale_key"], cache[2])
        self.assertIs(kwargs["dequant_scale_value"], cache[3])
        self.assertEqual((kwargs["query_quant_mode"], kwargs["key_quant_mode"], kwargs["value_quant_mode"]), (0, 4, 4))
        self.assertEqual(self.dynamic_quant.call_count, 2)
        for call in self.dynamic_quant.call_args_list:
            self.assertEqual(call.args[0].shape, (2, 256))
            self.assertEqual(call.kwargs["quant_mode"], "pertoken")
        restored_k, restored_v, ks, vs = impl.reshape_and_cache.call_args.args[:4]
        self.assertEqual(restored_k.shape, k.shape)
        self.assertEqual(restored_v.shape, v.shape)
        self.assertEqual(ks.shape, (2,))
        self.assertEqual(vs.shape, (2,))
        for original, data_cache, scale_cache in ((k, cache[0], cache[2]), (v, cache[1], cache[3])):
            expected_scale = original.float().abs().flatten(1).amax(-1) / 448
            expected_fp8 = (original.float() / expected_scale[:, None, None]).to(torch.float8_e4m3fn)
            torch.testing.assert_close(scale_cache[1, :2], expected_scale, rtol=0, atol=0)
            torch.testing.assert_close(data_cache[1, :2].float().view_as(original), expected_fp8.float())
        self.assertTrue(torch.isfinite(out).all())

    def test_decode_append_preserves_historical_data_and_scales(self):
        impl, layer, meta, cache, q, k, v, out = self.setup_forward()
        impl.forward(layer, q, k, v, cache, meta, out)
        history = [x[1, :2].view(torch.uint8).clone() for x in cache]
        meta.actual_seq_lengths_q, meta.seq_lens_list = [1], [3]
        meta.slot_mapping = torch.tensor([514], dtype=torch.int32)
        impl.forward(layer, q[:1], k[:1] * 100, v[:1] / 32, cache, meta, out[:1])
        for tensor, expected in zip(cache, history):
            torch.testing.assert_close(tensor[1, :2].view(torch.uint8), expected, rtol=0, atol=0)
        self.assertGreater(cache[2][1, 2], cache[2][1, 0] * 50)
        self.assertLess(cache[3][1, 2], cache[3][1, 0] / 16)

    def test_noncontiguous_input_and_page_crossing_keep_token_scale_alignment(self):
        cache, raw, spec = self.make_cache(padded=True, ratio=2)
        raw.fill_(69)
        count = 515
        source = torch.randn(count, 2, 256, dtype=torch.bfloat16)[:, :, ::2]
        self.assertFalse(source.is_contiguous())
        quant, scale = reference_quantize(
            source.reshape(count, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        # Physical page 3 holds the prefix, page 1 holds the last three tokens.
        slots = torch.cat((torch.arange(512) + 3 * 512, torch.arange(3) + 512))
        before = raw.clone()
        self.namespace["scatter_fia_fp8_cache"](
            quant.view_as(source), quant.view_as(source), scale, scale * 2, cache, slots
        )
        torch.testing.assert_close(cache[0][3].float(), quant[:512].float(), rtol=0, atol=0)
        torch.testing.assert_close(cache[2][1, :3], scale[-3:], rtol=0, atol=0)
        torch.testing.assert_close(cache[3][1, :3], scale[-3:] * 2, rtol=0, atol=0)
        pitch = spec.page_size_bytes // 2
        for page in (0, 2, 4, 5, 6, 7):
            torch.testing.assert_close(
                raw[page * pitch : (page + 1) * pitch], before[page * pitch : (page + 1) * pitch]
            )

    def test_multiple_requests_use_physical_slots_for_both_scales(self):
        impl, layer, meta, cache, q, k, v, out = self.setup_forward()
        meta.actual_seq_lengths_q, meta.seq_lens_list = [1, 2], [1, 1]
        meta.block_tables = torch.tensor([[2], [1]], dtype=torch.int32)
        meta.slot_mapping = torch.tensor([1024, 512], dtype=torch.int32)
        impl.forward(layer, q, k, v, cache, meta, out)
        scales = k.float().flatten(1).abs().amax(-1) / 448
        torch.testing.assert_close(cache[2][torch.tensor([2, 1]), 0], scales, rtol=0, atol=0)
        # Each request has one visible token: attention must return its V.
        quant, vs = reference_quantize(v.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        expected = (quant.float() * vs[:, None]).to(torch.bfloat16).view(2, 2, 128).repeat_interleave(3, dim=1)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_padding_updates_null_slot_for_data_and_scales(self):
        cache, _, _ = self.make_cache()
        payload = torch.ones(2, 2, 128).to(torch.float8_e4m3fn)
        ks, vs = torch.tensor([0.25, 0.5]), torch.tensor([2.0, 4.0])
        self.namespace["scatter_fia_fp8_cache"](payload, payload, ks, vs, cache, torch.tensor([-1, 513]))
        self.assertEqual(cache[0][-1, -1].float().sum().item(), 0)
        self.assertEqual(cache[0][1, 1].float().sum().item(), 256)
        self.assertEqual(cache[2][0, 0].item(), 0.25)
        self.assertEqual(cache[3][1, 1].item(), 4.0)

    def test_empty_step_does_not_quantize_write_or_run_attention(self):
        impl, layer, meta, cache, q, k, v, out = self.setup_forward()
        meta.actual_seq_lengths_q = [0]
        impl.forward(layer, q[:0], k[:0], v[:0], cache, meta, out[:0])
        self.dynamic_quant.assert_not_called()
        self.scatter.assert_not_called()
        self.fia.assert_not_called()

    def test_capture_reuses_scale_cache_and_sink_uses_original_q(self):
        impl, layer, meta, cache, q, _, _, out = self.setup_forward()
        layer.sink_key = torch.randn(2, 128, dtype=torch.bfloat16)
        self.context.capturing = True
        impl.forward(layer, q, None, None, cache, meta, out)
        op, kwargs, provider = self.register.call_args.args
        self.assertIs(op, self.fia.out)
        self.assertEqual(provider.layer_name, "layer")
        self.assertIs(kwargs["dequant_scale_key"], cache[2])
        self.assertIs(kwargs["dequant_scale_value"], cache[3])
        self.assertIs(kwargs["out"][0].untyped_storage(), out.untyped_storage())
        torch.testing.assert_close(self.sink.call_args.args[0], q, rtol=0, atol=0)
        self.dynamic_quant.assert_not_called()

    def test_pd_rejected_before_cache_write(self):
        impl, layer, meta, cache, q, k, v, out = self.setup_forward()
        impl.vllm_config.kv_transfer_config = object()
        with self.assertRaisesRegex(NotImplementedError, "PD"):
            impl.forward(layer, q, k, v, cache, meta, out)
        self.scatter.assert_not_called()

    def quant_method_namespace(self):
        namespace = {
            "torch": torch,
            "AscendAttentionScheme": object,
            "get_current_vllm_config": lambda: SimpleNamespace(
                kv_transfer_config=None, model_config=SimpleNamespace(dtype=torch.bfloat16)
            ),
        }
        load_nodes(
            self.root / "vllm_ascend/quantization/methods/kv_cache/mxfp_c8.py",
            namespace,
            (ast.FunctionDef, ast.ClassDef),
        )
        return namespace

    def test_fia_accepts_checkpoint_v_scale_without_using_static_scales(self):
        namespace = self.quant_method_namespace()
        method = namespace["AscendC8MXFPKVCacheAttentionMethod"]({}, "layer", "fia_pseudo")
        layer = torch.nn.Module()
        raw = torch.zeros(256, dtype=torch.uint8)
        layer.register_parameter("v_cache_scale", torch.nn.Parameter(raw, requires_grad=False))
        method.process_weights_after_loading(layer)
        self.assertEqual(tuple(layer.state_dict()), ("v_cache_scale",))
        self.assertFalse(hasattr(layer, "v_cache_scale_float_reciprocal"))
        self.assertEqual(layer.v_cache_scale.sum().item(), 0)

    def test_original_qfa_still_decodes_checkpoint_v_scale(self):
        namespace = self.quant_method_namespace()
        method = namespace["AscendC8MXFPKVCacheAttentionMethod"]({}, "layer")
        layer = torch.nn.Module()
        raw = torch.arange(256).remainder(5).add(120).to(torch.uint8)
        layer.register_parameter("v_cache_scale", torch.nn.Parameter(raw, requires_grad=False))
        method.process_weights_after_loading(layer)
        expected = torch.exp2(127 - raw.float()).to(torch.bfloat16)
        torch.testing.assert_close(layer.v_cache_scale_float_reciprocal, expected, rtol=0, atol=0)

    def test_quant_method_installs_selected_fia_backend(self):
        namespace = self.quant_method_namespace()
        modules = {}
        for suffix in ("attention_c8_mxfp", "attention_c8_fia"):
            name = "vllm_ascend.attention." + suffix
            module = ModuleType(name)
            for key, value in self.namespace.items():
                if key.startswith("AscendC8"):
                    setattr(module, key, value)
            modules[name] = module
        layer = torch.nn.Module()
        layer.num_kv_heads, layer.head_size_v = 2, 128
        layer.impl = self.namespace["AscendC8MXFPAttentionBackendImpl"]()
        method = namespace["AscendC8MXFPKVCacheAttentionMethod"]({}, "layer", "fia_pseudo")
        with patch.dict("sys.modules", modules):
            method.create_weights(layer)
        self.assertIs(layer.attn_backend, self.namespace["AscendC8FIAPseudoAttentionBackend"])
        self.assertIsInstance(layer.impl, self.namespace["AscendC8FIAPseudoAttentionBackendImpl"])

    def test_v1_zeroer_includes_both_dynamic_scale_caches(self):
        cache, _, spec = self.make_cache(padded=True, ratio=2)
        namespace = {
            "torch": torch,
            "KVBlockZeroer": object,
            "FullAttentionSpec": SimpleNamespace,
            "_component_views_share_slot": lambda *args: False,
            "_is_page_strided_mla_cache": lambda *args: False,
            "largest_power_of_2_divisor": lambda n: n & -n,
        }
        load_nodes(self.root / "vllm_ascend/worker/utils.py", namespace, (ast.ClassDef,), {"AscendKVBlockZeroer"})
        zeroer = namespace["AscendKVBlockZeroer"](torch.device("cpu"), False)
        group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["layer"])
        zeroer.init_meta([group], [512], "mxfp8", set(), {"layer": SimpleNamespace(kv_cache=cache)})
        addresses, sizes, _, _, count = zeroer._meta
        self.assertEqual(count, 8)  # Four components, two kernel blocks per scheduler page.
        self.assertIn(cache[3].data_ptr(), addresses.tolist())
        # Simulate the zeroer's address plan and check only scheduler page 1.
        for tensor in cache:
            tensor.fill_(1)
        raw = torch.empty(0, dtype=torch.int8).set_(
            cache[0].untyped_storage(), 0, (cache[0].untyped_storage().nbytes(),)
        )
        raw_base = raw.data_ptr()
        before = raw.clone()
        for address, size, stride in zip(addresses.tolist(), sizes.tolist(), zeroer._seg_page_strides.tolist()):
            offset = address - raw_base + stride * 4
            raw[offset : offset + size * 4].zero_()
        for tensor in cache:
            self.assertTrue((tensor[2:4].float() == 0).all())
            self.assertTrue((tensor[:2].float() == 1).all())
            self.assertTrue((tensor[4:].float() == 1).all())
        pitch = spec.page_size_bytes
        torch.testing.assert_close(raw[32 : 32 + pitch], before[32 : 32 + pitch])

    def test_v2_zeroer_preserves_only_static_uint8_v_scales(self):
        namespace = {"torch": torch, "SimpleNamespace": SimpleNamespace, "replace": replace, "KVBlockZeroer": object}
        delegate = Mock()
        load_nodes(self.root / "vllm_ascend/worker/v2/utils.py", namespace, (ast.ClassDef,), {"AscendV2KVBlockZeroer"})
        namespace["KVBlockZeroer"] = delegate
        group_type = make_dataclass("Group", [("layer_names", list)])
        for static, expected in ((False, 4), (True, 3)):
            with self.subTest(static=static):
                cache, raw, spec = self.make_cache()
                if static:
                    cache = self.namespace["mxfp_cache_views_for_spec"](raw, spec, 512)
                delegate.reset_mock()
                namespace["AscendV2KVBlockZeroer"](
                    device=torch.device("cpu"),
                    attn_groups_iter=[group_type(["layer"])],
                    kernel_block_sizes=[512],
                    static_forward_context={"layer": SimpleNamespace(kv_cache=cache)},
                    num_blocks=4,
                    cache_dtype="mxfp8",
                )
                self.assertEqual(delegate.call_count, expected)


if __name__ == "__main__":
    unittest.main()
