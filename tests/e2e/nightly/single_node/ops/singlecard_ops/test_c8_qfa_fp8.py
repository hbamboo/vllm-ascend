# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU QFA mode 6 producer, paged history, learned sink and graph replay.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. Production definitions are
loaded without initializing vLLM distributed groups. All quantizers, cache
writes and attention calls execute the installed Ascend 950 operators.
"""

import importlib.machinery
import importlib.util
import runpy
import sys
import types
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


def load_qfa_helpers():
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
    helpers = loader(torch_npu)

    # Import the actual installed QFA wrapper without the package __init__,
    # which eagerly JIT-builds unrelated operators in a standalone process.
    package = Path(importlib.util.find_spec("cann_ops_transformer").origin).parent
    parent = types.ModuleType("cann_ops_transformer")
    parent.__path__ = [str(package)]
    parent.__file__ = str(package / "__init__.py")
    parent.__spec__ = importlib.machinery.ModuleSpec("cann_ops_transformer", loader=None, is_package=True)
    sys.modules[parent.__name__] = parent
    name = "c8_qfa_fp8_native_api"
    spec = importlib.util.spec_from_file_location(name, package / "ops/attention/quant_flash_attn/quant_flash_attn.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    ops = types.ModuleType("cann_ops_transformer.ops")
    ops.__path__ = [str(package / "ops")]
    ops.quant_flash_attn = native.quant_flash_attn
    ops.quant_flash_attn_metadata = native.quant_flash_attn_metadata
    scatter_path = package / "ops/attention/scatter_pa_kv_cache_with_k_scale/scatter_pa_kv_cache_with_k_scale.py"
    scatter_spec = importlib.util.spec_from_file_location("c8_qfa_scatter_native_api", scatter_path)
    scatter = importlib.util.module_from_spec(scatter_spec)
    sys.modules[scatter_spec.name] = scatter
    scatter_spec.loader.exec_module(scatter)
    ops.scatter_pa_kv_cache_with_k_scale = scatter.scatter_pa_kv_cache_with_k_scale
    sys.modules[ops.__name__] = ops
    for name in ("qfa_v_head_scale", "qfa_kv_scatter", "sink"):
        load_triton_helper(root, name)
    return helpers, native


def load_triton_helper(root, name):
    import triton
    import triton.language as tl

    module_name = f"vllm_ascend.ops.triton.{name}"
    spec = importlib.util.spec_from_file_location(module_name, root / f"vllm_ascend/ops/triton/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Only bypass plugin-dependent initialization; the kernels execute the
    # installed compiler/runtime on the real NPU with their production tiles.
    with patch.dict(
        sys.modules,
        {
            "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
            "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=lambda: None),
        },
    ):
        spec.loader.exec_module(module)
    return module


class TestC8QFAFP8NPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        cls.helpers, cls.native = load_qfa_helpers()
        cls.checkpoint_v_scale = None
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_case(self, q_len, padding=0, kv_heads=1, *, static_per_channel=False, kv_len=529):
        torch.manual_seed(101)
        heads, dim, kernel_block_size = 6, 128, 128
        logical_block_size = 512
        physical_pages = list(range((kv_len + logical_block_size - 1) // logical_block_size, 0, -1))
        logical_blocks = len(physical_pages) + 1
        pitch = self.helpers.mxfp_packet_size_bytes(kv_heads, logical_block_size, dim, dim) // 4 + padding
        raw = torch.full((logical_blocks * 4 * pitch,), 23, dtype=torch.int8, device="npu")
        spec = SimpleNamespace(
            num_kv_heads=kv_heads,
            head_size=dim,
            head_size_v=dim,
            block_size=logical_block_size,
            page_size_bytes=pitch * 4,
        )
        cache = self.helpers.mxfp_cache_views_for_spec(raw, spec, kernel_block_size, qfa_fp8=True)
        query = (torch.randn(q_len, heads, dim) / 4).bfloat16().npu()
        magnitude = torch.exp2(torch.arange(kv_len).remainder(7).float() - 3).reshape(-1, 1, 1)
        key = (torch.randn(kv_len, kv_heads, dim) * magnitude).bfloat16().npu()
        value = (torch.randn(kv_len, kv_heads, dim) / 4).bfloat16().npu()
        value.mul_(torch.exp2(torch.arange(kv_heads, device="npu").float() * 3)[None, :, None])
        layer = SimpleNamespace()
        positions = torch.arange(kv_len, device="npu", dtype=torch.int64)
        page_ids = torch.tensor(physical_pages, dtype=torch.int64, device="npu")
        slots = page_ids[positions // logical_block_size] * logical_block_size + positions % logical_block_size
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=q_len,
            num_actual_tokens=q_len,
            actual_seq_lengths_q=[q_len],
            seq_lens_list=[kv_len],
            query_start_loc_gpu=torch.tensor([0, q_len], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([kv_len], dtype=torch.int32, device="npu"),
            block_tables=torch.tensor(
                [[4 * page + offset for page in physical_pages for offset in range(4)]], dtype=torch.int32, device="npu"
            ),
            attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu(),
        )
        producer_metadata = SimpleNamespace(**vars(metadata))
        producer_metadata.max_query_len = producer_metadata.num_actual_tokens = kv_len
        producer_metadata.actual_seq_lengths_q = [kv_len]
        producer_metadata.query_start_loc_gpu = torch.tensor([0, kv_len], dtype=torch.int32, device="npu")
        producer_metadata.seq_lens_gpu = metadata.seq_lens_gpu.clone()
        producer_metadata.slot_mapping = slots
        producer_query = query[:1].expand(kv_len, -1, -1).contiguous()
        producer_output = torch.empty_like(producer_query)
        impl = self.helpers.AscendC8MXFPAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, dim
        impl.scale, impl.sliding_window = dim**-0.5, None
        impl.kv_sharing_target_layer_name = None
        impl.q_quant_mode = "per_token"
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        tiny = torch.finfo(torch.float32).tiny
        impl.qfa_v_descale = torch.full((kv_heads,), tiny, dtype=torch.float32, device="npu")
        impl.qfa_v_head_bound = torch.full((kv_heads,), tiny * 448, dtype=torch.bfloat16, device="npu")
        impl.qfa_p_scale = torch.tensor([448.0], dtype=torch.float32, device="npu")
        if static_per_channel:
            from safetensors import safe_open

            if self.checkpoint_v_scale is None:
                checkpoint = Path("/mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8")
                weight_name = "model.layers.1.self_attn.fa_v.scale"
                for path in sorted(checkpoint.glob("*.safetensors")):
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        if weight_name in set(handle.keys()):
                            type(self).checkpoint_v_scale = handle.get_tensor(weight_name).flatten().clone()
                            print("CHECKPOINT_V_SCALE", str(path), weight_name, flush=True)
                            break
                assert self.checkpoint_v_scale is not None, "Target checkpoint V scale was not found"
            codes = self.checkpoint_v_scale.clone()
            if codes.numel() < kv_heads * dim:
                codes = codes.repeat((kv_heads * dim + codes.numel() - 1) // codes.numel())
            codes = codes[: kv_heads * dim]
            codes[codes == 0] = 127
            channel_scale = torch.exp2(codes.float() - 127).reshape(kv_heads, dim).npu()
            layer.v_cache_scale_float_reciprocal = channel_scale.reciprocal().flatten().bfloat16()
            impl.q_quant_mode = impl.k_quant_mode = "per_token_head"
            impl.qfa_v_descale.fill_(1)
            impl.qfa_v_output_scale = channel_scale
            query.mul_(torch.tensor([1 / 64, 1 / 8, 0.5, 2, 8, 32], device="npu")[None, :, None])
            key.mul_(torch.exp2(torch.arange(kv_heads, device="npu").float() * 6 - 3)[None, :, None])
        reference_v_bound = torch.zeros(kv_heads)
        output = torch.empty_like(query)
        return SimpleNamespace(**locals())

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

    def reference(self, case):
        stored_tokens = case.key.shape[0]
        q8, qs = torch_npu.npu_dynamic_quant(
            case.query if case.static_per_channel else case.query.flatten(1),
            dst_type=torch.float8_e4m3fn,
            quant_mode="pertoken",
        )
        k8, ks = torch_npu.npu_dynamic_quant(
            case.key if case.static_per_channel else case.key.flatten(1),
            dst_type=torch.float8_e4m3fn,
            quant_mode="pertoken",
        )
        if case.static_per_channel:
            v8 = torch_npu.npu_quantize(
                case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
            ).reshape_as(case.value)
            vs = case.channel_scale
        else:
            case.reference_v_bound = torch.maximum(
                case.reference_v_bound, case.value.cpu().float().abs().amax(dim=(0, 2))
            )
            bounded_v = torch.cat(
                (
                    case.value.permute(1, 0, 2).reshape(case.kv_heads, -1),
                    case.reference_v_bound.bfloat16().npu()[:, None].expand(-1, 32),
                ),
                dim=1,
            )
            v8, vs = torch_npu.npu_dynamic_quant(bounded_v, dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        q8, qs, k8, ks, v8, vs = (x.cpu() for x in (q8, qs, k8, ks, v8, vs))
        if case.static_per_channel:
            q = q8.float() * qs[..., None]
            k = k8.float() * ks[..., None]
            v = v8.float() * vs[None]
        else:
            v8 = (
                v8.view(torch.uint8)[:, : stored_tokens * case.dim]
                .reshape(case.kv_heads, stored_tokens, case.dim)
                .permute(1, 0, 2)
                .contiguous()
                .view(torch.float8_e4m3fn)
            )
            q = q8.reshape(case.q_len, case.heads, case.dim).float() * qs[:, None, None]
            k = k8.reshape(stored_tokens, case.kv_heads, case.dim).float() * ks[:, None, None]
            v = v8.float() * vs[None, :, None]
        group = case.heads // case.kv_heads
        scores = torch.einsum("thd,shd->hts", q, k[: case.kv_len].repeat_interleave(group, 1)) * case.impl.scale
        mask = torch.arange(case.kv_len)[None] > torch.arange(case.q_len)[:, None] + case.kv_len - case.q_len
        scores.masked_fill_(mask[None], float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v[: case.kv_len].repeat_interleave(group, 1))
        if hasattr(case.layer, "sink_key"):
            sink = case.layer.sink_key.cpu().repeat_interleave(group, dim=0)
            sink_logits = (q * sink[None]).sum(-1) * case.impl.scale
            expected *= torch.sigmoid(scores.logsumexp(-1).t() - sink_logits)[..., None]
        return expected, scores, (q8, qs, k8, ks, v8, vs)

    def check_case(self, case, *, expected_output=None):
        expected, _, (_, _, k8, ks, v8, vs) = self.reference(case)
        stored_tokens = case.key.shape[0]
        pages, offsets = case.slots.cpu() // 128, case.slots.cpu() % 128
        for cached, source in zip(case.cache[:2], (k8, v8)):
            gathered = cached.cpu().view(torch.uint8)[pages, :, offsets, :].reshape(stored_tokens, -1)
            torch.testing.assert_close(gathered, source.reshape(stored_tokens, -1).view(torch.uint8), rtol=0, atol=0)
        cached_scales = case.cache[2].cpu()[pages, :, offsets]
        expected_scales = ks if case.static_per_channel else ks[:, None].expand(-1, case.kv_heads)
        torch.testing.assert_close(cached_scales, expected_scales, rtol=0, atol=0)
        if case.static_per_channel:
            torch.testing.assert_close(case.impl.qfa_v_descale.cpu(), torch.ones(case.kv_heads), rtol=0, atol=0)
        else:
            torch.testing.assert_close(case.cache[3].cpu()[pages], vs[None].expand(stored_tokens, -1), rtol=0, atol=0)
            torch.testing.assert_close(case.impl.qfa_v_descale.cpu(), vs, rtol=0, atol=0)
        self.assertGreater(ks.unique().numel(), 1)
        payload_bytes = (2 * 128 * 128 + 128 * 4 + 4) * case.kv_heads
        self.assertTrue(torch.all(case.raw.cpu().reshape(-1, case.pitch)[:, payload_bytes:] == 23))
        if expected_output is not None:
            expected = expected_output
        torch.testing.assert_close(case.output.cpu(), expected.bfloat16(), rtol=0.03, atol=0.03)
        print(
            "QFA_FP8_PASS",
            "q_len",
            case.q_len,
            "kv_len",
            case.kv_len,
            "padding",
            case.padding,
            "max_error",
            (case.output.cpu().float() - expected).abs().max().item(),
            flush=True,
        )

    def test_prefill_decode_and_page_padding(self):
        for q_len in (1, 6, 129):
            for padding in (0, 256):
                with self.subTest(q_len=q_len, padding=padding):
                    case = self.make_case(q_len, padding)
                    self.run_case(case)
                    torch.npu.synchronize()
                    self.check_case(case)

    def test_multiple_kv_heads_use_independent_dynamic_v_scales(self):
        case = self.make_case(6, 256, kv_heads=2)
        self.run_case(case)
        torch.npu.synchronize()
        self.check_case(case)
        self.assertNotEqual(*case.impl.qfa_v_descale.cpu().tolist())

    def test_token_head_qk_with_checkpoint_v_and_optional_sink(self):
        for kv_heads, with_sink in ((1, False), (1, True), (2, False), (2, True)):
            with self.subTest(kv_heads=kv_heads, with_sink=with_sink):
                case = self.make_case(6, 256, kv_heads=kv_heads, static_per_channel=True)
                selected_heads = (0,) if kv_heads == 1 else (0, 3)
                _, scores, (q8, qs, _, ks, _, _) = self.reference(case)
                self.assertEqual(qs.shape, (6, 6))
                self.assertEqual(ks.shape, (529, kv_heads))
                self.assertTrue(torch.all(qs[:, 1:] != qs[:, :-1]))
                if kv_heads > 1:
                    self.assertTrue(torch.all(ks[:, 0] != ks[:, 1]))
                if with_sink:
                    q_hat = q8.float() * qs[..., None]
                    sink = torch.empty(kv_heads, 128)
                    for kv_head, head in enumerate(selected_heads):
                        direction = q_hat[0, head]
                        delta = case.query.cpu()[0, head].float() - direction
                        orthogonal = delta - direction * delta.dot(direction) / direction.square().sum()
                        sink[kv_head] = direction * scores[head, 0].logsumexp(-1) / (
                            direction.square().sum() * case.impl.scale
                        ) + orthogonal * 2 / (orthogonal.square().sum() * case.impl.scale)
                    case.layer.sink_key = sink.npu()
                self.run_case(case)
                torch.npu.synchronize()
                self.check_case(case)
                if with_sink:
                    expected, _, _ = self.reference(case)
                    for head in selected_heads:
                        error = (case.output.cpu()[0, head].float() - expected[0, head]).norm()
                        self.assertLess((error / expected[0, head].norm()).item(), 0.1)

                # A sharing consumer must use the owner's native descale and
                # checkpoint channel scale, even if its own fields differ.
                consumer = self.helpers.AscendC8MXFPAttentionBackendImpl()
                consumer.__dict__.update(case.impl.__dict__)
                consumer.kv_sharing_target_layer_name = "owner"
                consumer.qfa_v_descale = torch.full_like(case.impl.qfa_v_descale, 7)
                consumer.qfa_v_output_scale = torch.full_like(case.impl.qfa_v_output_scale, 123)
                consumer.vllm_config = SimpleNamespace(
                    kv_transfer_config=None,
                    compilation_config=SimpleNamespace(
                        static_forward_context={"owner": SimpleNamespace(impl=case.impl)}
                    ),
                )
                shared_output = torch.empty_like(case.output)
                consumer.forward(case.layer, case.query, None, None, case.cache, case.metadata, output=shared_output)
                torch.testing.assert_close(shared_output, case.output, rtol=0, atol=0)

    def test_token_head_static_v_scatter_above_4gib(self):
        case = self.make_case(6, 256, kv_heads=2, static_per_channel=True, kv_len=6)
        pages = (5 * 1024**3) // case.pitch
        raw = torch.zeros(pages * case.pitch, dtype=torch.int8, device="npu")
        case.cache = self.helpers.qfa_fp8_cache_views(raw, pages, case.spec, 128, page_stride_bytes=case.pitch)
        outputs = []
        key8, key_scale = torch_npu.npu_dynamic_quant(case.key, dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
        value8 = torch_npu.npu_quantize(
            case.value.flatten(1), case.layer.v_cache_scale_float_reciprocal, None, torch.float8_e4m3fn, -1, False
        ).reshape_as(case.value)
        for page in (4, pages - 4):
            outputs.append(
                self.forward_step(
                    case,
                    case.query,
                    case.key,
                    case.value,
                    [0, 6],
                    [6],
                    [[page]],
                    list(range(page * 128, page * 128 + 6)),
                ).cpu()
            )
            for cached, expected in zip(case.cache[:2], (key8, value8)):
                actual = cached[page, :, :6].view(torch.uint8).permute(1, 0, 2).contiguous().cpu()
                torch.testing.assert_close(actual, expected.view(torch.uint8).cpu(), rtol=0, atol=0)
            torch.testing.assert_close(case.cache[2][page, :, :6].t().cpu(), key_scale.cpu(), rtol=0, atol=0)
        self.assertGreater((pages - 4) * case.pitch, 2**32)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        print("QFA_64BIT_SCATTER_PASS", (pages - 4) * case.pitch, flush=True)

    def test_token_head_static_v_skips_negative_padding(self):
        control = self.make_case(6, kv_heads=2, static_per_channel=True)
        case = self.make_case(6, kv_heads=2, static_per_channel=True)
        valid_slots = list(range(8 * 128, 8 * 128 + 6))
        query, key, value = control.query, control.key[:6], control.value[:6]
        expected = self.forward_step(control, query, key, value, [0, 6], [6], [[8]], valid_slots)
        padded_query = torch.cat((query, query * 1000))
        padded_key = torch.cat((key, key * 1000))
        padded_value = torch.cat((value, torch.full_like(value, float("nan"))))
        before = case.raw.cpu().reshape(-1, case.pitch).clone()
        actual = self.forward_step(
            case,
            padded_query,
            padded_key,
            padded_value,
            [0, 6, 12],
            [6, 1],
            [[8], [0]],
            valid_slots + [-1] * 6,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual[:6], expected, rtol=0, atol=0)
        for cached, reference in zip(case.cache[:3], control.cache[:3]):
            cached = cached.view(torch.uint8) if cached.dtype == torch.float8_e4m3fn else cached
            reference = reference.view(torch.uint8) if reference.dtype == torch.float8_e4m3fn else reference
            torch.testing.assert_close(cached, reference, rtol=0, atol=0)
        untouched = torch.arange(before.shape[0]) != 8
        torch.testing.assert_close(case.raw.cpu().reshape(-1, case.pitch)[untouched], before[untouched], rtol=0, atol=0)

    def test_token_head_static_v_graph_refreshes_inputs_and_slots(self):
        case = self.make_case(6, 256, kv_heads=2, static_per_channel=True, kv_len=6)
        metadata = case.producer_metadata
        query, key, value = case.query, case.key, case.value
        output = torch.empty_like(query)

        def run():
            case.impl.forward(case.layer, query, key, value, case.cache, metadata, output=output)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        metadata.qfa_metadata_cache = {}
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        for q_factor, k_factor, v_factor, page in ((2, 0.25, 4, 5), (0.125, 8, 0.5, 4)):
            query.mul_(q_factor)
            key.mul_(k_factor)
            value.mul_(v_factor)
            metadata.block_tables[0, 0] = page
            metadata.slot_mapping.copy_(torch.arange(6, device="npu") + page * 128)
            graph.replay()
            torch.npu.synchronize()
            case.slots = metadata.slot_mapping
            case.output = output
            graph_output = output.clone()
            run()
            torch.npu.synchronize()
            graph_eager_equal = torch.equal(output, graph_output)
            torch.testing.assert_close(output, graph_output, rtol=0, atol=0)
            expected, scores, (_, _, _, _, v8, channel_scale) = self.reference(case)
            # This six-token case fits one QFA page. Independently model the
            # native P path: cast exp(logit-max)*448 to E4M3 for PV, while
            # retaining the unquantized softmax denominator. QFA emits BF16
            # before the fused epilogue restores checkpoint channel scales.
            exp_scores = (scores - scores.amax(-1, keepdim=True)).exp()
            p8 = (exp_scores * 448).to(torch.float8_e4m3fn).float()
            group = case.heads // case.kv_heads
            pv = torch.einsum("hts,shd->thd", p8, v8.float().repeat_interleave(group, dim=1))
            native_output = (pv / (exp_scores.sum(-1).t()[..., None] * 448)).bfloat16()
            p8_expected = (native_output.float() * channel_scale.repeat_interleave(group, dim=0)[None]).bfloat16()
            actual = graph_output.cpu().float()
            fp32_error = (actual - expected.bfloat16().float()).abs()
            worst_flat = fp32_error.argmax().item()
            row, column = divmod(worst_flat, case.heads * case.dim)
            head, channel = divmod(column, case.dim)
            worst = (row, head, channel)
            print(
                "QFA_STATIC_GRAPH_DIAGNOSTIC",
                {
                    "q_factor": q_factor,
                    "k_factor": k_factor,
                    "v_factor": v_factor,
                    "page": page,
                    "graph_eager_equal": graph_eager_equal,
                    "fp32_reference_max_abs": fp32_error.max().item(),
                    "fp8_p_reference_max_abs": (actual - p8_expected.float()).abs().max().item(),
                    "worst_index": worst,
                    "actual_at_worst": actual[worst].item(),
                    "fp32_reference_at_worst": expected.bfloat16()[worst].item(),
                    "fp8_p_reference_at_worst": p8_expected[worst].item(),
                },
                flush=True,
            )
            # The independent P8 model is valid here because KV=6 occupies
            # one QFA softmax block; it is not a multi-block golden.
            torch.testing.assert_close(graph_output.cpu(), p8_expected, rtol=0.008, atol=1e-4)
            self.check_case(case, expected_output=p8_expected)

    def test_token_head_static_v_graph_shares_lengths_and_plan_across_readers(self):
        case = self.make_case(6, 256, kv_heads=2, static_per_channel=True)
        self.run_case(case)
        shared = {}
        readers = [SimpleNamespace(**vars(case.metadata)) for _ in range(2)]
        impls = [self.helpers.AscendC8MXFPAttentionBackendImpl() for _ in readers]
        outputs = [torch.empty_like(case.query) for _ in readers]
        for metadata, impl in zip(readers, impls):
            metadata.qfa_metadata_cache = {}
            metadata.qfa_shared_metadata_cache = shared
            impl.__dict__.update(case.impl.__dict__)

        def run(metadata_list, output_list):
            for impl, metadata, output in zip(impls, metadata_list, output_list):
                impl.forward(case.layer, case.query, None, None, case.cache, metadata, output=output)

        with patch.object(
            sys.modules["cann_ops_transformer.ops"],
            "quant_flash_attn_metadata",
            wraps=self.native.quant_flash_attn_metadata,
        ) as metadata_op:
            for _ in range(3):
                run(readers, outputs)
            torch.npu.synchronize()
            shared.clear()
            for metadata in readers:
                metadata.qfa_metadata_cache = {}
            metadata_op.reset_mock()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run(readers, outputs)
            self.assertEqual(metadata_op.call_count, 1)
            previous = outputs[0].clone()
            original_query = case.query.clone()

            for step, (factor, used) in enumerate(((1.5, 513), (1.0, 529)), start=1):
                case.query.copy_(original_query * factor)
                case.metadata.seq_lens_gpu.fill_(used)
                case.kv_len = used
                graph.replay()
                torch.npu.synchronize()
                captured = [output.clone() for output in outputs]
                self.assertEqual(metadata_op.call_count, step)
                self.assertFalse(torch.equal(captured[0], previous))
                previous = captured[0]
                eager_shared = {}
                eager_metadata = [SimpleNamespace(**vars(metadata)) for metadata in readers]
                eager_outputs = [torch.empty_like(case.query) for _ in readers]
                for metadata in eager_metadata:
                    metadata.qfa_metadata_cache = {}
                    metadata.qfa_shared_metadata_cache = eager_shared
                run(eager_metadata, eager_outputs)
                torch.npu.synchronize()
                self.assertEqual(metadata_op.call_count, step + 1)
                for actual, expected in zip(captured, eager_outputs):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(*captured, rtol=0, atol=0)
                case.output = captured[0]
                self.check_case(case)
                print("QFA_SHARED_STATIC_GRAPH_PASS", used, flush=True)

    def test_token_head_static_v_prefill_6144_then_decode_6(self):
        case = self.make_case(6144, 256, kv_heads=2, static_per_channel=True, kv_len=6150)
        prefill_metadata = SimpleNamespace(**vars(case.producer_metadata))
        prefill_metadata.num_actual_tokens = prefill_metadata.max_query_len = 6144
        prefill_metadata.actual_seq_lengths_q = [6144]
        prefill_metadata.seq_lens_list = [6144]
        prefill_metadata.query_start_loc_gpu = torch.tensor([0, 6144], dtype=torch.int32, device="npu")
        prefill_metadata.seq_lens_gpu = torch.tensor([6144], dtype=torch.int32, device="npu")
        prefill_metadata.slot_mapping = case.slots[:6144]
        prefill_output = torch.empty_like(case.query)
        case.impl.forward(
            case.layer,
            case.query,
            case.key[:6144],
            case.value[:6144],
            case.cache,
            prefill_metadata,
            output=prefill_output,
        )
        # Check the final prefill rows against a full-history FP32 oracle,
        # avoiding a redundant 6144x6144 CPU attention matrix.
        reference_case = SimpleNamespace(**vars(case))
        reference_case.query = case.query[-6:]
        reference_case.q_len, reference_case.kv_len = 6, 6144
        expected, _, _ = self.reference(reference_case)
        torch.testing.assert_close(prefill_output[-6:].cpu(), expected.bfloat16(), rtol=0.03, atol=0.03)

        decode_query = case.query[-6:].clone() * 0.75
        decoded = self.forward_step(
            case,
            decode_query,
            case.key[-6:],
            case.value[-6:],
            [0, 6],
            [6150],
            case.metadata.block_tables.cpu().tolist(),
            case.slots[-6:].cpu().tolist(),
        )
        reference_case.query = decode_query
        reference_case.output = decoded
        reference_case.kv_len = 6150
        self.check_case(reference_case)

    def forward_step(self, case, query, key, value, cu_q, seq_lens, block_tables, slots):
        metadata = SimpleNamespace(
            causal=True,
            max_query_len=max(end - start for start, end in zip(cu_q, cu_q[1:])),
            num_actual_tokens=query.shape[0],
            actual_seq_lengths_q=cu_q[1:],
            seq_lens_list=seq_lens,
            query_start_loc_gpu=torch.tensor(cu_q, dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor(seq_lens, dtype=torch.int32, device="npu"),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device="npu"),
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device="npu"),
            attn_mask=case.metadata.attn_mask,
        )
        output = torch.empty_like(query)
        case.impl.forward(case.layer, query, key, value, case.cache, metadata, output=output)
        return output

    def test_v_scale_growth_preserves_history_and_ignores_new_page_garbage(self):
        case = self.make_case(1)
        case.cache[3][8].fill_(float("nan"))
        before = case.raw.cpu().reshape(-1, case.pitch).clone()
        query = torch.zeros(1, 6, 128, dtype=torch.bfloat16, device="npu")
        key = torch.zeros(1, 1, 128, dtype=torch.bfloat16, device="npu")
        value = torch.ones_like(key)
        first = self.forward_step(case, query, key, value, [0, 1], [1], [[8]], [8 * 128])
        torch.testing.assert_close(first.cpu(), torch.ones_like(first.cpu()), rtol=0.01, atol=0.01)
        second = self.forward_step(case, query, key, value * 100, [0, 1], [2], [[8]], [8 * 128 + 1])
        torch.npu.synchronize()
        torch.testing.assert_close(second.cpu(), torch.full_like(second.cpu(), 50.5), rtol=0.01, atol=0.02)
        historical_value = case.cache[1].cpu()[8, 0, 0].float() * case.impl.qfa_v_descale.cpu()[0]
        torch.testing.assert_close(historical_value, torch.ones(128), rtol=0.02, atol=0.002)
        untouched = torch.arange(before.shape[0]) != 8
        torch.testing.assert_close(case.raw.cpu().reshape(-1, case.pitch)[untouched], before[untouched], rtol=0, atol=0)

    def test_shared_prefix_rescales_once_and_unreferenced_pages_stay_untouched(self):
        case = self.make_case(1)
        before = case.raw.cpu().reshape(-1, case.pitch).clone()
        prefix_query = torch.zeros(128, 6, 128, dtype=torch.bfloat16, device="npu")
        prefix_key = torch.zeros(128, 1, 128, dtype=torch.bfloat16, device="npu")
        self.forward_step(
            case,
            prefix_query,
            prefix_key,
            torch.ones_like(prefix_key),
            [0, 128],
            [128],
            [[8]],
            list(range(8 * 128, 9 * 128)),
        )
        query = torch.zeros(2, 6, 128, dtype=torch.bfloat16, device="npu")
        key = torch.zeros(2, 1, 128, dtype=torch.bfloat16, device="npu")
        value = torch.tensor([100.0, 50.0], dtype=torch.bfloat16, device="npu")[:, None, None].expand(2, 1, 128)
        result = self.forward_step(
            case, query, key, value, [0, 1, 2], [129, 129], [[8, 9], [8, 10]], [9 * 128, 10 * 128]
        )
        torch.npu.synchronize()
        old_value = case.cache[1].cpu()[8, 0].float() * case.impl.qfa_v_descale.cpu()[0]
        torch.testing.assert_close(old_value, torch.ones_like(old_value), rtol=0.02, atol=0.002)
        expected = torch.stack(((old_value.sum(0) + 100) / 129, (old_value.sum(0) + 50) / 129))
        expected = expected[:, None, :].expand(2, 6, 128).bfloat16()
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.02)
        untouched = ~torch.isin(torch.arange(before.shape[0]), torch.tensor([8, 9, 10]))
        torch.testing.assert_close(case.raw.cpu().reshape(-1, case.pitch)[untouched], before[untouched], rtol=0, atol=0)

    def test_history_marking_skips_stale_zero_query_rows(self):
        helper = sys.modules["vllm_ascend.ops.triton.qfa_v_head_scale"]
        history = helper.qfa_v_history_pages(
            torch.tensor([[8, 9], [8, 10], [1, 2]], dtype=torch.int32, device="npu"),
            torch.tensor([0, 1, 2, 2], dtype=torch.int32, device="npu"),
            torch.tensor([129, 129, 128], dtype=torch.int32, device="npu"),
            12,
        )
        expected = torch.zeros(12, dtype=torch.int32)
        expected[8], expected[9], expected[10] = 129, 1, 1
        torch.testing.assert_close(history.cpu(), expected, rtol=0, atol=0)

    def test_invalid_graph_rows_do_not_change_v_head_quantization(self):
        query = torch.zeros(12, 6, 128, dtype=torch.bfloat16, device="npu")
        key = torch.zeros(12, 1, 128, dtype=torch.bfloat16, device="npu")
        valid_slots = list(range(8 * 128, 8 * 128 + 6))
        control = self.make_case(6)
        expected = self.forward_step(
            control, query[:6], key[:6], torch.ones_like(key[:6]), [0, 6], [6], [[8]], valid_slots
        )
        for padded_value, invalid_slot in ((1e20, -1), (float("nan"), -1), (1e20, 12 * 128 + 7)):
            with self.subTest(padded_value=padded_value, invalid_slot=invalid_slot):
                case = self.make_case(6)
                value = torch.ones_like(key)
                value[6:].fill_(padded_value)
                actual = self.forward_step(
                    case,
                    query,
                    key,
                    value,
                    [0, 6, 12],
                    [6, 1],
                    [[8], [0]],
                    valid_slots + [invalid_slot] * 6,
                )
                torch.npu.synchronize()
                torch.testing.assert_close(actual[:6], expected, rtol=0, atol=0)
                for cached, reference in zip(case.cache[:3], control.cache[:3]):
                    actual_payload, expected_payload = cached[8, :, :6], reference[8, :, :6]
                    if cached.dtype == torch.float8_e4m3fn:
                        actual_payload = actual_payload.view(torch.uint8)
                        expected_payload = expected_payload.view(torch.uint8)
                    torch.testing.assert_close(actual_payload, expected_payload, rtol=0, atol=0)
                torch.testing.assert_close(case.cache[3][8], control.cache[3][8], rtol=0, atol=0)
                torch.testing.assert_close(case.impl.qfa_v_descale, control.impl.qfa_v_descale, rtol=0, atol=0)
                torch.testing.assert_close(case.impl.qfa_v_head_bound, control.impl.qfa_v_head_bound, rtol=0, atol=0)

    def test_graph_replay_masks_invalid_v_before_head_max_reduction(self):
        case = self.make_case(6)
        query = torch.zeros(12, 6, 128, dtype=torch.bfloat16, device="npu")
        key = torch.zeros(12, 1, 128, dtype=torch.bfloat16, device="npu")
        value = torch.ones_like(key)
        value[6:].fill_(1e4)
        valid_slots = list(range(8 * 128, 8 * 128 + 6))
        metadata = SimpleNamespace(**vars(case.metadata))
        metadata.num_actual_tokens = 12
        metadata.actual_seq_lengths_q = [6, 12]
        metadata.seq_lens_list = [6, 1]
        metadata.query_start_loc_gpu = torch.tensor([0, 6, 12], dtype=torch.int32, device="npu")
        metadata.seq_lens_gpu = torch.tensor([6, 1], dtype=torch.int32, device="npu")
        metadata.block_tables = torch.tensor([[8], [0]], dtype=torch.int32, device="npu")
        metadata.slot_mapping = torch.tensor(valid_slots + [-1] * 6, dtype=torch.int64, device="npu")
        output = torch.empty_like(query)
        scale_address = case.impl.qfa_v_descale.data_ptr()
        before = case.raw.cpu().reshape(-1, case.pitch).clone()

        def run():
            case.impl.forward(case.layer, query, key, value, case.cache, metadata, output=output)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        metadata.qfa_metadata_cache = {}
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        baseline = output[:6].clone()
        baseline_payload = case.cache[1][8, :, :6].view(torch.uint8).clone()
        baseline_scale = case.impl.qfa_v_descale.clone()
        for padding in (float("nan"), 1e20):
            value[6:].fill_(padding)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output[:6], baseline, rtol=0, atol=0)
            torch.testing.assert_close(case.cache[1][8, :, :6].view(torch.uint8), baseline_payload, rtol=0, atol=0)
            torch.testing.assert_close(case.impl.qfa_v_descale, baseline_scale, rtol=0, atol=0)
            torch.testing.assert_close(
                case.impl.qfa_v_head_bound.cpu(), torch.ones(1, dtype=torch.bfloat16), rtol=0, atol=0
            )

        # The same row now owns a valid cache slot, so its V=100 must grow
        # the shared head bound. Other padded values remain excluded.
        value[6].fill_(100)
        metadata.slot_mapping[6] = 8 * 128 + 6
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(case.impl.qfa_v_head_bound.cpu(), torch.full((1,), 100, dtype=torch.bfloat16))
        reconstructed = case.cache[1][8, 0, 6].float() * case.impl.qfa_v_descale[0]
        torch.testing.assert_close(reconstructed.cpu(), torch.full((128,), 100.0), rtol=0.01, atol=0.01)
        grown_scale = case.impl.qfa_v_descale.clone()
        metadata.slot_mapping[6] = -1
        value[6:].fill_(1e30)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(case.impl.qfa_v_descale, grown_scale, rtol=0, atol=0)
        self.assertEqual(case.impl.qfa_v_descale.data_ptr(), scale_address)
        untouched = ~torch.isin(torch.arange(before.shape[0]), torch.tensor([0, 8]))
        torch.testing.assert_close(case.raw.cpu().reshape(-1, case.pitch)[untouched], before[untouched], rtol=0, atol=0)

    def test_all_pad_dummy_preserves_initial_v_scale_and_other_pages(self):
        case = self.make_case(6)
        initial_scale = case.impl.qfa_v_descale.clone()
        initial_bound = case.impl.qfa_v_head_bound.clone()
        before = case.raw.cpu().reshape(-1, case.pitch).clone()
        query = torch.zeros(12, 6, 128, dtype=torch.bfloat16, device="npu")
        key = torch.zeros(12, 1, 128, dtype=torch.bfloat16, device="npu")
        value = torch.full_like(key, float("nan"))
        self.forward_step(case, query, key, value, [0, 6, 12], [1, 1], [[0], [0]], [-1] * 12)
        torch.npu.synchronize()
        torch.testing.assert_close(case.impl.qfa_v_descale, initial_scale, rtol=0, atol=0)
        torch.testing.assert_close(case.impl.qfa_v_head_bound, initial_bound, rtol=0, atol=0)
        # Dummy metadata may update page 0's tag, but every usable page is
        # untouched and invalid NaNs cannot enter the persistent head bound.
        torch.testing.assert_close(case.raw.cpu().reshape(-1, case.pitch)[1:], before[1:], rtol=0, atol=0)

    def test_lse_and_learned_key_sink_use_quantized_query(self):
        for q_len in (1, 6):
            with self.subTest(q_len=q_len):
                case = self.make_case(q_len)
                _, scores, (q8, qs, _, _, _, _) = self.reference(case)
                q_hat = q8.reshape(q_len, 6, 128).float() * qs[:, None, None]
                direction = q_hat[0, 0]
                delta = case.query.cpu()[0, 0].float() - direction
                orthogonal = delta - direction * delta.dot(direction) / direction.square().sum()
                # The selected head has factor ~0.5. A component orthogonal
                # to Qhat makes use of raw Q change its sink logit by ~2,
                # so this also detects an inconsistent sink query producer.
                sink = direction * scores[0, 0].logsumexp(-1) / (direction.square().sum() * case.impl.scale)
                sink += orthogonal * 2 / (orthogonal.square().sum() * case.impl.scale)
                case.layer.sink_key = sink[None].npu()
                self.run_case(case)
                self.check_case(case)
                expected, scores, (q8, qs, _, _, _, _) = self.reference(case)
                relative_error = (case.output.cpu()[0, 0].float() - expected[0, 0]).norm() / expected[0, 0].norm()
                self.assertLess(relative_error.item(), 0.08)
                q_ntd = q8.reshape(q_len, 6, 128).view(torch.uint8).permute(1, 0, 2).contiguous().npu()
                q_ntd = q_ntd.view(torch.float8_e4m3fn)
                mask_mode = 0 if q_len == 1 else 3
                plans = case.metadata.qfa_metadata_cache
                key = ("fp8_token", 6, 1, 128, q_len, mask_mode)
                _, lse = self.native.quant_flash_attn(
                    q_ntd,
                    case.cache[0],
                    case.cache[1],
                    qs[None].expand(6, -1).contiguous().npu(),
                    case.cache[2],
                    case.impl.qfa_v_descale,
                    6,
                    p_scale=case.impl.qfa_p_scale,
                    block_table=case.metadata.block_tables,
                    cu_seqlens_q=case.metadata.query_start_loc_gpu,
                    seqused_kv=case.metadata.seq_lens_gpu,
                    metadata=plans[key],
                    softmax_scale=case.impl.scale,
                    mask_mode=mask_mode,
                    attn_mask=case.metadata.attn_mask if mask_mode else None,
                    max_seqlen_q=q_len,
                    max_seqlen_kv=-1,
                    layout_q="NTD",
                    layout_q_descale="NT",
                    layout_kv="PA_BNBD",
                    layout_out="TND",
                    return_softmax_lse=True,
                )
                torch.testing.assert_close(lse.cpu().reshape(6, q_len), scores.logsumexp(-1), rtol=1e-4, atol=1e-4)

    def test_graph_replay_refreshes_lengths_and_dynamic_scales(self):
        case = self.make_case(1, 256)
        scale_address = case.impl.qfa_v_descale.data_ptr()
        for _ in range(3):
            self.run_case(case)
        self.assertEqual(case.impl.qfa_v_descale.data_ptr(), scale_address)
        torch.npu.synchronize()
        case.metadata.qfa_metadata_cache = {}
        case.producer_metadata.qfa_metadata_cache = {}
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.run_case(case)
        for factor, kv_len in ((2.0, 513), (0.125, 529), (16.0, 513)):
            case.key.mul_(factor)
            case.query.mul_(1 / factor)
            case.value.mul_(factor)
            case.kv_len = kv_len
            case.metadata.seq_lens_gpu.fill_(kv_len)
            graph.replay()
            torch.npu.synchronize()
            self.assertEqual(case.impl.qfa_v_descale.data_ptr(), scale_address)
            self.check_case(case)
            graph_output = case.output.clone()
            self.run_case(case)
            torch.npu.synchronize()
            self.assertEqual(case.impl.qfa_v_descale.data_ptr(), scale_address)
            torch.testing.assert_close(case.output, graph_output, rtol=0, atol=0)

    def test_key_extrema_are_not_clipped_to_mx_power_of_two_range(self):
        case = self.make_case(1)
        row = torch.tensor([1.875, 1.0, 0.5, -1.875], dtype=torch.bfloat16, device="npu").repeat(32)
        case.key.copy_(row.reshape(1, 1, 128))
        self.run_case(case)
        torch.npu.synchronize()
        reconstructed = case.cache[0].cpu()[8, 0, 0].float() * case.cache[2].cpu()[8, 0, 0]
        torch.testing.assert_close(reconstructed[[0, 3]].bfloat16(), row.cpu()[[0, 3]], rtol=0, atol=0)


class TestQFASinkNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        cls.sink = load_triton_helper(root, "sink")

    def make_case(self, tokens, heads, kv_heads, *, strided=False, sink_dtype=torch.float32):
        torch.manual_seed(91)
        dim = 128
        query = torch.randn(tokens, heads, dim, device="npu", dtype=torch.bfloat16)
        quantized, query_scale = torch_npu.npu_dynamic_quant(
            query.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        query8 = quantized.reshape_as(query)
        if strided:
            storage = torch.empty(tokens, heads + 1, dim, dtype=torch.uint8, device="npu")
            storage[:, :heads].copy_(query8.view(torch.uint8))
            query8 = storage[:, :heads].view(torch.float8_e4m3fn)
        attn = torch.randn(tokens, heads, dim, device="npu", dtype=torch.bfloat16)
        lse = torch.randn(heads, tokens + 3, device="npu")[:, :tokens]
        sink_key = torch.randn(kv_heads, dim, device="npu", dtype=sink_dtype)
        destination = torch.full((tokens + 3, heads, dim), 23, dtype=torch.bfloat16, device="npu")
        output = destination[:tokens]
        scale = dim**-0.5
        return SimpleNamespace(**locals())

    def run_case(self, case):
        self.sink.qfa_sink_apply_triton(
            case.query8,
            case.query_scale,
            case.attn,
            case.lse,
            case.sink_key,
            case.scale,
            case.tokens,
            case.heads,
            case.kv_heads,
            case.dim,
            case.output,
        )

    def check_case(self, case, before):
        q_hat = case.query8.cpu().float() * case.query_scale.cpu()[:, None, None]
        sink = case.sink_key.cpu().float().repeat_interleave(case.heads // case.kv_heads, dim=0)
        logit = (q_hat * sink[None]).sum(-1) * case.scale
        factor = torch.sigmoid(case.lse.cpu().t() - logit)
        expected = (case.attn.cpu().float() * factor[..., None]).bfloat16()
        torch.testing.assert_close(case.output.cpu(), expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(case.attn, before, rtol=0, atol=0)
        self.assertTrue(torch.all(case.destination[case.tokens :].cpu() == 23))

    def test_fp8_query_head_major_lse_and_direct_output(self):
        for tokens, heads, kv_heads, strided, sink_dtype in (
            (1, 6, 1, False, torch.float32),
            (6, 6, 1, True, torch.bfloat16),
            (129, 8, 2, True, torch.float32),
            (6144, 6, 1, False, torch.float32),
        ):
            with self.subTest(tokens=tokens, heads=heads, kv_heads=kv_heads, strided=strided):
                case = self.make_case(tokens, heads, kv_heads, strided=strided, sink_dtype=sink_dtype)
                before = case.attn.clone()
                self.run_case(case)
                torch.npu.synchronize()
                self.check_case(case, before)

    def test_graph_replay_uses_updated_scale_and_lse(self):
        case = self.make_case(6, 6, 1, strided=True)
        before = case.attn.clone()
        for _ in range(3):
            self.run_case(case)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.run_case(case)
        case.query_scale.mul_(2)
        case.lse.add_(1)
        graph.replay()
        torch.npu.synchronize()
        self.check_case(case, before)


if __name__ == "__main__":
    unittest.main()
