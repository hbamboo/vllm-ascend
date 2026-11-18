# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone real-NPU K-block SWA backend and UpdatableGraph contracts.

From the repository root, run with ASCEND_RT_VISIBLE_DEVICES=0 and PYTHONPATH=.
  python tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_kblock_attention.py -v

Production definitions are loaded without initializing vLLM platform plugins.
Only the metadata builder's parent is a minimal pass-through: the new builder
itself performs real NPU copies into its fixed buffers. Full-service tests
must separately validate the parent metadata and runner integration.
"""

import ast
import importlib.util
import runpy
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exec_definitions(module, path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), module.__dict__)


class _MetadataParent:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.max_num_blocks_per_req = 10

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata


class TestC8KBlockAttentionNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        import triton
        import triton.language as tl

        root = next(path for path in Path(__file__).resolve().parents if (path / "vllm_ascend").is_dir())
        loader = runpy.run_path(str(root / "tests/ut/attention/test_c8_k_per_token.py"))["load_helpers"]
        cls.helpers = loader(torch_npu)
        weak = ModuleType("c8_kblock_test_weak_refs")
        weak.__dict__.update(torch=torch, torch_npu=torch_npu, IntermediateTensors=type("IntermediateTensors", (), {}))
        _exec_definitions(weak, root / "vllm_ascend/utils.py", {"weak_ref_tensor", "weak_ref_tensors"})
        cls.imports = patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
                "vllm.logger": SimpleNamespace(logger=SimpleNamespace(debug_once=lambda *args, **kwargs: None)),
                "vllm_ascend.utils": weak,
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=lambda: None),
            },
        )
        cls.imports.start()
        cls.addClassCleanup(cls.imports.stop)
        cls.kernel = _load_module(
            "vllm_ascend.ops.triton.c8_kblock_cache", root / "vllm_ascend/ops/triton/c8_kblock_cache.py"
        )
        sink = _load_module("c8_kblock_test_sink", root / "vllm_ascend/ops/triton/sink.py")
        cls.graph_module = _load_module("c8_kblock_test_graph", root / "vllm_ascend/compilation/updatable_graph.py")
        backend = ModuleType("c8_kblock_test_backend")
        backend.__dict__.update(vars(cls.helpers))
        backend.__dict__.update(
            dataclass=dataclass,
            field=field,
            AscendMetadata=object,
            AscendC8MXFPMetadataBuilder=_MetadataParent,
            weak_ref_tensors=weak.weak_ref_tensors,
            sink_apply_triton=sink.sink_apply_triton,
            get_capture_resource=cls.graph_module.get_capture_resource,
            register_task=cls.graph_module.register_task,
            _FIA_V2_WORKSPACE_KEY="npu_fused_infer_attention_score_v2.workspace",
        )
        sys.modules[backend.__name__] = backend
        _exec_definitions(
            backend,
            root / "vllm_ascend/attention/attention_c8_kblock.py",
            {
                "AscendC8KBlockMetadata",
                "AscendC8KBlockMetadataBuilder",
                "KBlockFIAParamProvider",
                "AscendC8KBlockAttentionBackendImpl",
            },
        )
        cls.backend = backend
        cls.config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=32))
        cls.mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
        torch.set_num_threads(4)
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def make_frame(self, name, window, pages):
        spec = SimpleNamespace(num_kv_heads=1, head_size=128, head_size_v=128)
        pitch = sum(self.helpers.fia_kblock_channel_packet_section_sizes(1, 512, 128, 32)) + 64
        raw = torch.full((pages * pitch,), 90, dtype=torch.int8, device="npu")
        cache = self.helpers.fia_kblock_channel_cache_views(raw, pages, spec, 512, 32, page_stride_bytes=pitch)
        impl = self.backend.AscendC8KBlockAttentionBackendImpl()
        impl.num_heads, impl.num_kv_heads, impl.head_size = 6, 1, 128
        impl.scale, impl.sliding_window, impl.k_block_group_size = 128**-0.5, window, 32
        impl.kv_sharing_target_layer_name = None
        impl.vllm_config = self.config
        impl._use_max_workspace_for_fia_graph = False
        generator = torch.Generator().manual_seed(91)
        layer = SimpleNamespace(layer_name=name, sink_key=(torch.randn(1, 128, generator=generator) * 2).npu())
        return SimpleNamespace(raw=raw, cache=cache, impl=impl, layer=layer, pitch=pitch)

    def batch(self, query_lens, seq_lens, seed, kind="strong"):
        generator = torch.Generator().manual_seed(seed)
        table = torch.randperm(len(query_lens) * 10, generator=generator).reshape(len(query_lens), 10).int() + 1
        query = torch.randn(sum(query_lens), 6, 128, generator=generator)
        query = (query / query.square().mean(-1, keepdim=True).add(1e-6).sqrt()).bfloat16()
        keys, values = [], []
        for query_len, seq_len in zip(query_lens, seq_lens):
            key = torch.randn(seq_len, 1, 128, generator=generator).bfloat16()
            key[..., 3] *= 6
            value = torch.randn(seq_len, 1, 128, generator=generator).bfloat16()
            if kind == "zero":
                query.zero_()
                key.zero_()
                value.zero_()
            elif kind == "boundary":
                query.zero_()
                value.fill_(0.25)
                left = max(0, seq_len - query_len - self.window)
                value[:left].fill_(1024)
                value[left] = 512
            keys.append(key)
            values.append(value)
        starts = [0]
        for length in query_lens:
            starts.append(starts[-1] + length)
        slots = []
        for row, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
            positions = torch.arange(seq_len - query_len, seq_len)
            slots.append(table[row, positions // 512].long() * 512 + positions % 512)
        return SimpleNamespace(
            query_lens=query_lens,
            seq_lens=seq_lens,
            starts=starts,
            table=table,
            query=query,
            keys=keys,
            values=values,
            key_input=torch.cat([key[-count:] for key, count in zip(keys, query_lens)]),
            value_input=torch.cat([value[-count:] for value, count in zip(values, query_lens)]),
            slots=torch.cat(slots),
            kind=kind,
        )

    def populate_history(self, frame, batch):
        frame.raw.fill_(90)
        counts = [length - count for length, count in zip(batch.seq_lens, batch.query_lens)]
        if not sum(counts):
            return
        starts, slots = [0], []
        for row, count in enumerate(counts):
            starts.append(starts[-1] + count)
            positions = torch.arange(count)
            slots.append(batch.table[row, positions // 512].long() * 512 + positions % 512)
        self.kernel.update_kblock_cache(
            torch.cat([key[:count] for key, count in zip(batch.keys, counts)]).npu(),
            torch.cat([value[:count] for value, count in zip(batch.values, counts)]).npu(),
            frame.cache,
            torch.cat(slots).npu(),
            torch.tensor(starts, dtype=torch.int32, device="npu"),
            torch.tensor(counts, dtype=torch.int32, device="npu"),
            batch.table.npu(),
            num_reqs=len(counts),
            active_num_reqs=torch.tensor([len(counts)], dtype=torch.int32, device="npu"),
            group_size=32,
            max_query_len=max(counts),
        )

    def metadata(self, builder, batch, slot_buffer=None):
        source = SimpleNamespace(
            actual_seq_lengths_q=batch.starts[1:],
            seq_lens_list=list(batch.seq_lens),
            query_start_loc_gpu=torch.tensor(batch.starts, dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor(batch.seq_lens, dtype=torch.int32, device="npu"),
            block_tables=batch.table.npu(),
            slot_mapping=batch.slots.npu() if slot_buffer is None else slot_buffer,
            attn_mask=self.mask,
            qfa_metadata_cache={},
            num_actual_tokens=batch.starts[-1],
            max_query_len=max(batch.query_lens),
        )
        return builder.build(0, source)

    def assert_oracle(self, frame, batch, actual, real_requests=None):
        # Decode the actual FP8 bytes with FP32 scales before an independent
        # CPU BF16-KV attention. Quantization itself has separate kernel UTs.
        key_bytes, value_bytes = (cache.view(torch.uint8).cpu() for cache in frame.cache[:2])
        key_scales, value_scales = (cache.cpu() for cache in frame.cache[2:4])
        sink = frame.layer.sink_key.cpu().bfloat16().float()
        real_requests = len(batch.seq_lens) if real_requests is None else real_requests
        errors = []
        for row in range(real_requests):
            count, length = batch.query_lens[row], batch.seq_lens[row]
            positions = torch.arange(length)
            pages, offsets = batch.table[row, positions // 512].long(), positions % 512
            key = key_bytes[pages, offsets].contiguous().view(torch.float8_e4m3fn).float().reshape(length, 1, 128)
            value = value_bytes[pages, offsets].contiguous().view(torch.float8_e4m3fn).float().reshape(length, 1, 128)
            key = (key * key_scales[pages, offsets // 32]).bfloat16().float().expand(-1, 6, -1)
            value = (value * value_scales[pages, offsets, :, None]).bfloat16().float().expand(-1, 6, -1)
            query = batch.query[batch.starts[row] : batch.starts[row + 1]].float()
            logits = torch.einsum("thd,shd->hts", query, key) * frame.impl.scale
            query_positions = torch.arange(length - count, length)
            hidden = (positions[None] > query_positions[:, None]) | (
                positions[None] < query_positions[:, None] - frame.impl.sliding_window
            )
            logits.masked_fill_(hidden[None], float("-inf"))
            expected = torch.einsum("hts,shd->thd", logits.softmax(-1), value).bfloat16()
            sink_logit = (query * sink).sum(-1).bfloat16().float() * frame.impl.scale
            factor = torch.sigmoid(logits.logsumexp(-1).t() - sink_logit).bfloat16()
            expected = (expected.float() * factor[..., None].float()).bfloat16()
            got = actual[batch.starts[row] : batch.starts[row + 1]].cpu()
            self.assertTrue(bool(torch.isfinite(got).all()))
            torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
            errors.append((got.float() - expected.float()).abs().max().item())
        print("ORACLE_PASS", batch.query_lens, batch.seq_lens, frame.impl.sliding_window, max(errors), flush=True)

    def test_prefill_decode_and_window_boundaries(self):
        for count, length, window, kind in (
            (529, 529, 64, "boundary"),
            (529, 529, 4096, "strong"),
            (1, 1, 64, "zero"),
            (1, 529, 64, "boundary"),
            (6, 529, 64, "strong"),
            (6, 4101, 4096, "boundary"),
            (6, 4102, 4096, "boundary"),
            (6, 4103, 4096, "boundary"),
            (6, 4231, 4096, "strong"),
        ):
            with self.subTest(query_len=count, seq_len=length, window=window, kind=kind):
                self.window = window
                batch = self.batch([count], [length], 700 + length, kind)
                frame = self.make_frame("swa", window, 11)
                self.populate_history(frame, batch)
                builder = self.backend.AscendC8KBlockMetadataBuilder(None, ["swa"], self.config, "npu:0")
                metadata = self.metadata(builder, batch)
                query, key, value = batch.query.npu(), batch.key_input.npu(), batch.value_input.npu()
                output = torch.empty_like(query)
                sink = self.backend.sink_apply_triton
                seen = []

                def sink_spy(original_query, *args, seen=seen, sink=sink):
                    seen.append(original_query.cpu().clone())
                    return sink(original_query, *args)

                with patch.object(self.backend, "sink_apply_triton", sink_spy):
                    frame.impl.forward(frame.layer, query, key, value, frame.cache, metadata, output=output)
                self.assertEqual(len(seen), 1)
                torch.testing.assert_close(seen[0], batch.query, rtol=0, atol=0)
                torch.testing.assert_close(query.cpu(), batch.query, rtol=0, atol=0)
                torch.testing.assert_close(key.cpu(), batch.key_input, rtol=0, atol=0)
                torch.testing.assert_close(value.cpu(), batch.value_input, rtol=0, atol=0)
                provider = self.backend.KBlockFIAParamProvider("swa", window, torch.zeros(1, 10, dtype=torch.int32))
                self.assertEqual(provider.resolve({"swa": metadata})["actual_seq_kvlen"], [min(length, window + count)])
                self.assert_oracle(frame, batch, output)

    def test_mixed_single_and_six_query_requests(self):
        for window in (64, 4096):
            with self.subTest(window=window):
                self.window = window
                batch = self.batch([1, 6], [529, 4231], 801)
                frame = self.make_frame("mixed", window, 21)
                self.populate_history(frame, batch)
                builder = self.backend.AscendC8KBlockMetadataBuilder(None, ["mixed"], self.config, "npu:0")
                metadata = self.metadata(builder, batch)
                output = torch.empty((7, 6, 128), dtype=torch.bfloat16, device="npu")
                frame.impl.forward(
                    frame.layer,
                    batch.query.npu(),
                    batch.key_input.npu(),
                    batch.value_input.npu(),
                    frame.cache,
                    metadata,
                    output=output,
                )
                self.assert_oracle(frame, batch, output)

    def test_dummy_queries_exceed_dummy_kv_and_have_no_slots(self):
        self.window = 64
        batch = self.batch([2], [2], 901)
        batch.query = torch.cat((batch.query, torch.randn(3, 6, 128).bfloat16()))
        batch.key_input = torch.cat((batch.key_input, torch.full((3, 1, 128), 19, dtype=torch.bfloat16)))
        batch.value_input = torch.cat((batch.value_input, torch.full((3, 1, 128), 23, dtype=torch.bfloat16)))
        batch.query_lens, batch.seq_lens, batch.starts = [2, 3], [2, 1], [0, 2, 5]
        batch.table = torch.cat((batch.table, torch.zeros_like(batch.table)))
        frame = self.make_frame("dummy", 64, 11)
        builder = self.backend.AscendC8KBlockMetadataBuilder(None, ["dummy"], self.config, "npu:0")
        source = SimpleNamespace(
            actual_seq_lengths_q=[2, 5],
            seq_lens_list=[2, 1],
            query_start_loc_gpu=torch.tensor([0, 2, 5], dtype=torch.int32, device="npu"),
            seq_lens_gpu=torch.tensor([2], dtype=torch.int32, device="npu"),
            block_tables=batch.table.npu(),
            slot_mapping=batch.slots.npu(),
            attn_mask=self.mask,
            qfa_metadata_cache={},
            num_actual_tokens=2,
            max_query_len=3,
        )
        metadata = builder.build(0, source)
        output = torch.empty_like(batch.query, device="npu")
        frame.impl.forward(
            frame.layer,
            batch.query.npu(),
            batch.key_input.npu(),
            batch.value_input.npu(),
            frame.cache,
            metadata,
            output=output,
        )
        self.assertEqual(metadata.kblock_num_reqs.cpu().tolist(), [2])
        self.assertEqual(metadata.seq_lens_gpu[:2].cpu().tolist(), [2, 1])
        self.assert_oracle(frame, batch, output, real_requests=1)
        pages = frame.raw.cpu().view(11, frame.pitch)
        real_page = int(batch.table[0, 0])
        for page in range(11):
            if page != real_page:
                self.assertTrue(bool((pages[page] == 90).all()))
        # Dummy Q>K rows may be completely masked. Only real outputs must be finite.

    def test_full_updatable_graph_regroups_fixed_token_bucket(self):
        self.window = 4096
        frames = [self.make_frame(name, 4096, 321) for name in ("swa_a", "swa_b")]
        for frame in frames:
            frame.query = torch.empty((32, 6, 128), dtype=torch.bfloat16, device="npu")
            frame.key = torch.empty((32, 1, 128), dtype=torch.bfloat16, device="npu")
            frame.value = torch.empty_like(frame.key)
            frame.output = torch.empty_like(frame.query)
        slots = torch.empty(32, dtype=torch.int64, device="npu")
        builder = self.backend.AscendC8KBlockMetadataBuilder(None, ["swa_a", "swa_b"], self.config, "npu:0")

        def prepare(stage, counts, lengths):
            batches, snapshots = [], []
            for index, frame in enumerate(frames):
                batch = self.batch(counts, lengths, 1001 + stage * 71 + index)
                # One attention group has one runner-owned block table; both
                # layers use its mapping but retain distinct K/V activations.
                if batches:
                    batch.table = batches[0].table
                    batch.slots = batches[0].slots
                self.populate_history(frame, batch)
                frame.query.copy_(batch.query.npu())
                frame.key.copy_(batch.key_input.npu())
                frame.value.copy_(batch.value_input.npu())
                snapshots.append(frame.raw.clone())
                batches.append(batch)
            slots.copy_(batches[0].slots.npu())
            metadata = self.metadata(builder, batches[0], slots)
            torch.npu.synchronize()
            return batches, snapshots, metadata

        stages = (
            ([1] * 32, [1, 65, 513, 4097, 4231, 31, 511, 4102] * 4),
            ([4] * 8, [4, 64, 514, 4099, 4231, 33, 512, 4100]),
            ([1] * 32, [4230, 1, 4103, 514, 66, 32, 4096, 513] * 4),
        )
        _, initial, metadata = prepare(0, *stages[0])
        context = self.backend._EXTRA_CTX
        with patch.object(context, "capturing", True):
            for _ in range(3):
                for frame, snapshot in zip(frames, initial):
                    frame.raw.copy_(snapshot)
                    frame.impl.forward(
                        frame.layer, frame.query, frame.key, frame.value, frame.cache, metadata, output=frame.output
                    )
        torch.npu.synchronize()
        graph = self.graph_module.UpdatableGraph()
        update_stream = torch.npu.Stream()
        with patch.object(context, "capturing", True), torch.npu.graph(graph):
            for frame in frames:
                frame.impl.forward(
                    frame.layer, frame.query, frame.key, frame.value, frame.cache, metadata, output=frame.output
                )
        self.assertEqual(len(graph.tasks), 2)
        self.assertEqual(len(graph.capture_resources), 2)  # shared window tuple + FIA workspace
        for name in ("key", "value"):
            self.assertEqual(graph.tasks[0].kwargs[name].data_ptr(), graph.tasks[1].kwargs[name].data_ptr())
        pointers = tuple(
            t.data_ptr()
            for t in (
                builder.kblock_query_start_loc,
                builder.kblock_seq_lens,
                builder.kblock_block_tables,
                builder.kblock_num_reqs,
                slots,
                *(tensor for frame in frames for tensor in (frame.query, frame.key, frame.value)),
            )
        )
        for stage, (counts, lengths) in enumerate(stages):
            batches, snapshots, metadata = prepare(stage, counts, lengths)
            self.assertEqual(metadata.kblock_num_reqs.cpu().tolist(), [len(counts)])
            self.assertEqual(
                pointers,
                tuple(
                    t.data_ptr()
                    for t in (
                        builder.kblock_query_start_loc,
                        builder.kblock_seq_lens,
                        builder.kblock_block_tables,
                        builder.kblock_num_reqs,
                        slots,
                        *(tensor for frame in frames for tensor in (frame.query, frame.key, frame.value)),
                    )
                ),
            )
            resolved = graph.resolve_tasks(
                self.graph_module.ContextSource({frame.layer.layer_name: metadata for frame in frames})
            )
            for task in resolved:
                self.assertEqual(task.kwargs["actual_seq_qlen"], metadata.actual_seq_lengths_q)
                self.assertEqual(task.kwargs["actual_seq_kvlen"], [min(n, 4096 + q) for n, q in zip(lengths, counts)])
            update_stream.wait_stream(torch.npu.current_stream())
            graph.replay()
            graph.update(update_stream, resolved)
            torch.npu.synchronize()
            replay_outputs = [frame.output.cpu().clone() for frame in frames]
            replay_caches = [frame.raw.cpu().clone() for frame in frames]
            for frame, batch, snapshot, replay_output, replay_cache in zip(
                frames, batches, snapshots, replay_outputs, replay_caches
            ):
                # Both executions start from the same original history/stage,
                # never from the graph's already updated cache.
                frame.raw.copy_(snapshot)
                frame.output.fill_(-123)
                frame.impl.forward(
                    frame.layer, frame.query, frame.key, frame.value, frame.cache, metadata, output=frame.output
                )
                torch.npu.synchronize()
                self.assertTrue(torch.equal(frame.output.cpu().view(torch.uint8), replay_output.view(torch.uint8)))
                self.assertTrue(torch.equal(frame.raw.cpu(), replay_cache))
                self.assert_oracle(frame, batch, replay_output)
            print("UPDATABLE_GRAPH_PASS", stage, "T32", "B", len(counts), "output_and_raw_bytes_equal", flush=True)


if __name__ == "__main__":
    unittest.main()
