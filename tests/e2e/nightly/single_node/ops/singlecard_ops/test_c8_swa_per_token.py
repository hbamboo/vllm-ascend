# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real-NPU sliding-window attention with dynamic per-token FP8 KV.

Run directly with ASCEND_RT_VISIBLE_DEVICES=0. Production forward writes
the paged cache; the CPU oracle independently consumes the quantized KV.
"""

import importlib.util
import math
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


class TestC8SWAPerTokenNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
        cls.helpers = loader(torch_npu)
        import triton
        import triton.language as tl

        spec = importlib.util.spec_from_file_location("c8_swa_sink", root / "vllm_ascend/ops/triton/sink.py")
        sink = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=lambda: None),
            },
        ):
            spec.loader.exec_module(sink)
        cls.helpers.AscendC8FIAPseudoAttentionBackendImpl.forward.__globals__["sink_apply_triton"] = (
            sink.sink_apply_triton
        )
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_case(self, q_len=6, window=64, kv_len=529):
        torch.manual_seed(173)
        heads, dim, block_size = 6, 128, 512
        pitch = self.helpers.mxfp_packet_size_bytes(1, block_size, dim, dim) + 256
        raw = torch.zeros(3 * pitch, dtype=torch.int8, device="npu")
        cache = self.helpers.fia_pseudo_cache_views(
            raw,
            3,
            SimpleNamespace(num_kv_heads=1, head_size=dim, head_size_v=dim),
            block_size,
            page_stride_bytes=pitch,
        )
        token = torch.arange(kv_len)
        # Q=0 makes visibility and the inclusive left boundary independently
        # observable: one boundary V spike must contribute to the first row,
        # then disappear as the window advances. Old V strongly differs from
        # visible V, so silently taking full attention cannot pass.
        key = ((torch.randn(kv_len, 1, dim) / 8) + 0.25).bfloat16().npu()
        values = torch.full((kv_len, 1, dim), 0.25)
        first_query_position = kv_len - q_len
        values[: max(0, first_query_position - window)] = 32
        values[max(0, first_query_position - window)] = 128
        values[-q_len:] = 4 + token[-q_len:, None, None].remainder(3)
        value = values.bfloat16().npu()
        query = torch.zeros(q_len, heads, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
        positions = torch.arange(kv_len, device="npu", dtype=torch.int64)
        slots = torch.where(positions < block_size, positions + 2 * block_size, positions)
        block_table = torch.tensor([[2, 1]], dtype=torch.int32, device="npu")
        impl = self.helpers.AscendC8FIAPseudoAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = heads, 1, dim, dim**-0.5
        impl.sliding_window, impl.kv_sharing_target_layer_name = window, None
        layer = SimpleNamespace(layer_name="swa_attention")
        output = torch.empty_like(query)
        return SimpleNamespace(**locals())

    def metadata(self, case, q_len, kv_len, slots):
        return SimpleNamespace(
            causal=True,
            actual_seq_lengths_q=[q_len],
            seq_lens_list=[kv_len],
            slot_mapping=slots,
            block_tables=case.block_table,
            attn_mask=case.mask,
            num_actual_tokens=q_len,
            max_query_len=q_len,
            query_start_loc_gpu=torch.tensor([0, q_len], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([kv_len], dtype=torch.int32, device="npu"),
        )

    def populate(self, case):
        # Write two successive chunks, rather than pre-filling the cache
        # directly, so tail writes must preserve the first chunk's scales.
        for start in range(0, case.kv_len, case.block_size):
            end = min(start + case.block_size, case.kv_len)
            query = torch.zeros(end - start, case.heads, case.dim, dtype=torch.bfloat16, device="npu")
            metadata = self.metadata(case, end - start, end, case.slots[start:end])
            case.impl.forward(
                case.layer,
                query,
                case.key[start:end],
                case.value[start:end],
                case.cache,
                metadata,
                output=torch.empty_like(query),
            )

    def check_case(self, case):
        quantized = []
        for source, payload_cache, scale_cache in zip((case.key, case.value), case.cache[:2], case.cache[2:]):
            payload, scale = torch_npu.npu_dynamic_quant(
                source.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
            )
            payload, scale = payload.cpu(), scale.cpu()
            slots = case.slots.cpu()
            pages, offsets = slots // case.block_size, slots % case.block_size
            torch.testing.assert_close(
                payload_cache.cpu().view(torch.uint8)[pages, offsets], payload.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(scale_cache.cpu()[pages, offsets], scale, rtol=0, atol=0)
            quantized.append((payload.float() * scale[:, None]).bfloat16().float())
        key, value = (x.reshape(case.kv_len, 1, case.dim).repeat_interleave(case.heads, 1) for x in quantized)
        query = case.query.cpu().float()
        scores = torch.einsum("thd,shd->hts", query, key) / math.sqrt(case.dim)
        query_position = torch.arange(case.q_len) + case.kv_len - case.q_len
        key_position = torch.arange(case.kv_len)
        future = key_position[None] > query_position[:, None]
        old = key_position[None] < query_position[:, None] - case.window
        full_output = torch.einsum("hts,shd->thd", scores.masked_fill(future[None], float("-inf")).softmax(-1), value)
        scores.masked_fill_((future | old)[None], float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), value).bfloat16()
        if hasattr(case.layer, "sink_key"):
            sink = case.layer.sink_key.cpu().bfloat16().float()
            logit = ((query * sink).sum(-1)).bfloat16().float() * case.impl.scale
            factor = torch.sigmoid(scores.logsumexp(-1).t() - logit).bfloat16()
            expected = (expected.float() * factor[..., None]).bfloat16()
        actual = case.output.cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        self.assertTrue(torch.isfinite(actual).all())
        if case.kv_len > case.window + case.q_len:
            self.assertGreater((full_output - expected.float()).abs().max().item(), 0.5)
        print(
            "SWA_PASS",
            "q_len",
            case.q_len,
            "window",
            case.window,
            "kv_len",
            case.kv_len,
            "max_error",
            (actual - expected).abs().max().item(),
            flush=True,
        )

    def read(self, case):
        metadata = self.metadata(case, case.q_len, case.kv_len, case.slots[-case.q_len :])
        case.impl.forward(case.layer, case.query, None, None, case.cache, metadata, output=case.output)

    def test_window_and_page_boundaries(self):
        for q_len, window, kv_len in ((1, 64, 1), (1, 64, 529), (6, 64, 529), (128, 64, 529), (6, 512, 529)):
            with self.subTest(q_len=q_len, window=window, kv_len=kv_len):
                case = self.make_case(q_len, window, kv_len)
                self.populate(case)
                self.read(case)
                self.check_case(case)

    def test_mixed_single_and_multi_query_sequences(self):
        single = self.make_case(q_len=1)
        multi = self.make_case(q_len=6, kv_len=77)
        multi.value.mul_(0.25)
        multi.query.fill_(0.125)
        raw = torch.zeros(6 * single.pitch, dtype=torch.int8, device="npu")
        cache = self.helpers.fia_pseudo_cache_views(
            raw,
            6,
            SimpleNamespace(num_kv_heads=1, head_size=single.dim, head_size_v=single.dim),
            single.block_size,
            page_stride_bytes=single.pitch,
        )
        single.cache = multi.cache = cache
        multi.slots.add_(3 * multi.block_size)
        multi.block_table.add_(3)
        self.populate(single)
        self.populate(multi)
        metadata = self.metadata(single, 7, single.kv_len, torch.cat((single.slots[-1:], multi.slots[-6:])))
        metadata.actual_seq_lengths_q = [1, 7]
        metadata.seq_lens_list = [single.kv_len, multi.kv_len]
        metadata.block_tables = torch.cat((single.block_table, multi.block_table))
        metadata.query_start_loc_gpu = torch.tensor([0, 1, 7], dtype=torch.int32, device="npu")
        metadata.seq_lens_gpu = torch.tensor(metadata.seq_lens_list, dtype=torch.int32, device="npu")
        metadata.max_query_len = 6
        query = torch.cat((single.query, multi.query))
        output = torch.empty_like(query)
        single.impl.forward(single.layer, query, None, None, cache, metadata, output=output)
        single.output, multi.output = output[:1], output[1:]
        self.check_case(single)
        self.check_case(multi)

    def test_raw_graph_replay_refreshes_written_kv_and_scales(self):
        # Capture the production eager forward as device graph operations.
        # This does not exercise vLLM's register_task/UpdatableGraph path;
        # host lengths stay fixed while device KV and scale values change.
        case = self.make_case(q_len=1)
        case.query.fill_(0.125)
        self.populate(case)
        metadata = self.metadata(case, case.q_len, case.kv_len, case.slots[-case.q_len :])

        def step():
            case.impl.forward(
                case.layer,
                case.query,
                case.key[-case.q_len :],
                case.value[-case.q_len :],
                case.cache,
                metadata,
                output=case.output,
            )

        for _ in range(3):
            step()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            step()
        for factor in (4, 0.25):
            case.key[-case.q_len :].mul_(factor)
            case.value[-case.q_len :].mul_(factor)
            graph.replay()
            torch.npu.synchronize()
            self.check_case(case)
            replay_output = case.output.clone()
            step()
            torch.npu.synchronize()
            torch.testing.assert_close(case.output, replay_output, rtol=0, atol=0)

    def test_updatable_graph_refreshes_host_kv_lengths(self):
        # Initialize vLLM before importing the Ascend graph module to avoid
        # entering platform discovery from a partially imported plugin.
        import vllm  # noqa: F401

        from vllm_ascend.compilation.updatable_graph import (
            ContextSource,
            UpdatableGraph,
            get_capture_resource,
            register_task,
        )

        full = self.make_case(q_len=1)
        full.impl.sliding_window = None
        full.layer.layer_name = "full_attention"
        case = self.make_case(q_len=1)
        self.populate(full)
        self.populate(case)
        for _ in range(3):
            self.read(full)
            self.read(case)
        torch.npu.synchronize()
        full_metadata = self.metadata(full, 1, full.kv_len, full.slots[-1:])
        metadata = self.metadata(case, 1, case.kv_len, case.slots[-1:])
        full_key, full_value, full_slots = case.key, case.value, case.slots
        namespace = case.impl.forward.__globals__
        full.impl._use_max_workspace_for_fia_graph = False
        case.impl._use_max_workspace_for_fia_graph = False
        graph = UpdatableGraph()
        update_stream = torch.npu.Stream()
        with (
            patch.dict(
                namespace,
                {
                    "get_capture_resource": get_capture_resource,
                    "register_task": register_task,
                    "_FIA_V2_WORKSPACE_KEY": "npu_fused_infer_attention_score_v2.workspace",
                },
            ),
            patch.object(namespace["_EXTRA_CTX"], "capturing", True),
        ):
            with torch.npu.graph(graph):
                full.impl.forward(full.layer, full.query, None, None, full.cache, full_metadata, output=full.output)
                case.impl.forward(case.layer, case.query, None, None, case.cache, metadata, output=case.output)
            self.assertEqual(len(graph.tasks), 2)
            self.assertEqual(len(graph.capture_resources), 2)
            self.assertFalse(graph.tasks[0].provider.pad_single_query)
            self.assertTrue(graph.tasks[1].provider.pad_single_query)
            for kv_len in (513, 529):
                full_metadata.seq_lens_list = [kv_len]
                metadata.seq_lens_list = [kv_len]
                resolved = graph.resolve_tasks(
                    ContextSource({full.layer.layer_name: full_metadata, case.layer.layer_name: metadata})
                )
                self.assertEqual(resolved[0].kwargs["actual_seq_qlen"], [1])
                self.assertEqual(resolved[1].kwargs["actual_seq_qlen"], [2])
                for task in resolved:
                    self.assertEqual(task.kwargs["actual_seq_kvlen"], [kv_len])
                # Match the V2 model runner's external-event replay/update
                # ordering, rather than replacing task registration with a mock.
                update_stream.wait_stream(torch.npu.current_stream())
                graph.replay()
                graph.update(update_stream, resolved)
                torch.npu.synchronize()
                case.kv_len = kv_len
                case.key, case.value, case.slots = full_key[:kv_len], full_value[:kv_len], full_slots[:kv_len]
                self.check_case(case)
                full_output = full.output.clone()
                eager_output = torch.empty_like(full.output)
                with patch.object(namespace["_EXTRA_CTX"], "capturing", False):
                    full.impl.forward(
                        full.layer, full.query, None, None, full.cache, full_metadata, output=eager_output
                    )
                torch.npu.synchronize()
                torch.testing.assert_close(full_output, eager_output, rtol=0, atol=0)
                self.assertGreater((full_output - case.output).abs().max().item(), 1)
                print("FULL_GRAPH_PASS", "kv_len", kv_len, "max_error", 0, flush=True)

    def test_zeroed_page_reused_by_a_short_request(self):
        case = self.make_case()
        self.populate(case)
        other_page = case.raw.view(3, case.pitch)[1].clone()
        case.raw.view(3, case.pitch)[2].zero_()
        case.kv_len = 49
        case.key = (case.key[:49] * 2).contiguous()
        case.value = (torch.ones_like(case.value[:49]) * 0.75).contiguous()
        case.slots = case.slots[:49]
        metadata = self.metadata(case, case.kv_len, case.kv_len, case.slots)
        query = torch.zeros(case.kv_len, case.heads, case.dim, dtype=torch.bfloat16, device="npu")
        case.impl.forward(case.layer, query, case.key, case.value, case.cache, metadata, output=torch.empty_like(query))
        self.read(case)
        self.check_case(case)
        torch.testing.assert_close(case.raw.view(3, case.pitch)[1], other_page, rtol=0, atol=0)

    def test_learned_key_sink_uses_windowed_lse(self):
        case = self.make_case()
        case.query.fill_(0.125)
        case.layer.sink_key = torch.full((1, case.dim), 2, dtype=torch.float32, device="npu")
        self.populate(case)
        self.read(case)
        self.check_case(case)


if __name__ == "__main__":
    unittest.main()
