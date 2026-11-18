# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone Ascend 950 tests of real Triton V-scale history and graph replay."""

import sys
import unittest
from pathlib import Path

# The neighboring triton/ tests must not shadow the installed compiler.
if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())))

import torch
import torch_npu

from vllm_ascend.ops.triton.qfa_v_head_scale import (
    prepare_qfa_v_head_input,
    qfa_v_history_pages,
    rescale_qfa_v_cache,
)

NUM_PAGES = 4
NUM_HEADS = 2
BLOCK_SIZE = 128
HEAD_DIM = 128
FP8_MAX = 448.0
SENTINEL_SIZE = 32
KV_BYTES = NUM_HEADS * BLOCK_SIZE * HEAD_DIM
K_SCALE_BYTES = NUM_HEADS * BLOCK_SIZE * 4
PAGE_PITCH = 2 * KV_BYTES + K_SCALE_BYTES + NUM_HEADS * 4 + 64


def cache_views(raw):
    values = torch.as_strided(
        raw.view(torch.float8_e4m3fn),
        (NUM_PAGES, NUM_HEADS, BLOCK_SIZE, HEAD_DIM),
        (PAGE_PITCH, BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1),
        storage_offset=KV_BYTES,
    )
    scales = torch.as_strided(
        raw.view(torch.float32),
        (NUM_PAGES, NUM_HEADS),
        (PAGE_PITCH // 4, 1),
        storage_offset=(2 * KV_BYTES + K_SCALE_BYTES) // 4,
    )
    return values, scales


def make_backing():
    raw = torch.full((NUM_PAGES * PAGE_PITCH,), 55, dtype=torch.int8)
    values, scales = cache_views(raw)
    source = torch.arange(values.numel(), dtype=torch.float32).reshape(values.shape).remainder(29).sub(14).div(4)
    values.copy_(source.to(torch.float8_e4m3fn))
    # Fresh page 2 deliberately has no valid FP8 payload or scale to read.
    values[2].view(torch.uint8).fill_(127)
    scales.copy_(torch.tensor([[float("nan"), float("nan")], [0.5, 1.0], [float("nan"), float("nan")], [0.25, 2.0]]))
    return raw


def expected_backing(initial, codes, new_scale):
    raw = initial.clone()
    values, scales = cache_views(raw)
    for page, code in enumerate(codes):
        if code > 0:
            old_count = code - 1
            if old_count > 0:
                result = values[page, :, :old_count].float() * scales[page, :, None, None] / new_scale[:, None, None]
                values[page, :, :old_count].copy_(result.to(torch.float8_e4m3fn))
            scales[page].copy_(new_scale)
    return raw


def quantize_head_input(value, slots, bound):
    prepared = prepare_qfa_v_head_input(value, slots, bound, value.shape[0], NUM_PAGES * BLOCK_SIZE)
    payload, scale = torch_npu.npu_dynamic_quant(prepared, dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
    return prepared, payload, scale


class TestQfaVHeadScaleNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("FP8 cache tests require an Ascend 950 NPU")
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise unittest.SkipTest("FP8 cache tests require an Ascend 950 NPU")

    def setUp(self):
        self.initial = make_backing()
        self.raw = self.initial.to("npu")
        self.values, self.scales = cache_views(self.raw)
        self.block_tables = torch.tensor([[1, 2, 0, -1], [1, 3, 0, -1], [0, 0, 0, -1]], dtype=torch.int32, device="npu")
        self.cu_q = torch.tensor([0, 2, 3, 3], dtype=torch.int32, device="npu")
        # The stale third row has a nonzero sequence length but no scheduled Q.
        self.seq_lens = torch.tensor([130, 130, 256], dtype=torch.int32, device="npu")
        self.new_scale = torch.tensor([1.0, 4.0], dtype=torch.float32, device="npu")
        self.bound = torch.zeros(NUM_HEADS, dtype=torch.bfloat16, device="npu")
        self.head_scale = torch.zeros(NUM_HEADS, dtype=torch.float32, device="npu")

    def run_helpers(self):
        codes = qfa_v_history_pages(self.block_tables, self.cu_q, self.seq_lens, NUM_PAGES)
        rescale_qfa_v_cache(self.values, self.scales, self.new_scale, codes, self.bound, self.head_scale)
        return codes

    def test_shared_prefix_fresh_pages_and_unreferenced_bytes(self):
        codes = self.run_helpers()
        expected_codes = [0, 129, 1, 2]
        torch.testing.assert_close(codes.cpu(), torch.tensor(expected_codes, dtype=torch.int32))
        expected = expected_backing(self.initial, expected_codes, self.new_scale.cpu())
        # Comparing all bytes also covers K, K scales, padding, unrelated V
        # rows, the fresh page's NaNs, and the unreferenced physical page 0.
        torch.testing.assert_close(self.raw.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(self.bound.cpu(), (self.new_scale.cpu() * FP8_MAX).to(torch.bfloat16))
        torch.testing.assert_close(self.head_scale.cpu(), self.new_scale.cpu())

    def test_prepare_masks_padding_outliers_before_dynamic_quant(self):
        num_tokens = 6
        source = torch.linspace(-2, 2, num_tokens * NUM_HEADS * HEAD_DIM).reshape(num_tokens, NUM_HEADS, HEAD_DIM)
        clean = source.to(torch.bfloat16)
        clean[[1, 2, 4]] = 0
        dirty = clean.clone()
        dirty[1].fill_(1e20)
        dirty[2].fill_(float("nan"))
        dirty[4].fill_(-1e20)
        # Keep a padded source head stride, as produced by sliced projections.
        clean_storage = torch.empty((num_tokens, NUM_HEADS, HEAD_DIM + 16), dtype=torch.bfloat16, device="npu")
        dirty_storage = torch.empty_like(clean_storage)
        clean_npu = clean_storage[..., :HEAD_DIM]
        dirty_npu = dirty_storage[..., :HEAD_DIM]
        clean_npu.copy_(clean)
        dirty_npu.copy_(dirty)
        slots = torch.tensor([0, -1, NUM_PAGES * BLOCK_SIZE, 129, -1, 130], dtype=torch.int32, device="npu")
        bound = torch.tensor([0.5, 1.0], dtype=torch.bfloat16, device="npu")

        prepared, payload, scale = quantize_head_input(clean_npu, slots, bound)
        dirty_prepared, dirty_payload, dirty_scale = quantize_head_input(dirty_npu, slots, bound)
        expected = torch.cat(
            (clean.permute(1, 0, 2).reshape(NUM_HEADS, -1), bound.cpu()[:, None].expand(-1, SENTINEL_SIZE)), dim=1
        )
        torch.testing.assert_close(prepared.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(dirty_prepared.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(payload.view(torch.uint8).cpu(), dirty_payload.view(torch.uint8).cpu())
        torch.testing.assert_close(scale.cpu(), dirty_scale.cpu(), rtol=0, atol=0)

    def test_all_invalid_tokens_preserve_tiny_bound_and_scale(self):
        tiny = torch.finfo(torch.float32).tiny
        bound_cpu = torch.tensor([tiny * FP8_MAX, 2 * tiny * FP8_MAX], dtype=torch.bfloat16)
        self.bound.copy_(bound_cpu)
        self.head_scale.copy_(bound_cpu.float() / FP8_MAX)
        value = torch.full((6, NUM_HEADS, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device="npu")
        slots = torch.full((6,), -1, dtype=torch.int32, device="npu")
        prepared, payload, scale = quantize_head_input(value, slots, self.bound)
        torch.testing.assert_close(prepared[:, : 6 * HEAD_DIM].cpu(), torch.zeros(NUM_HEADS, 6 * HEAD_DIM).bfloat16())
        torch.testing.assert_close(
            payload[:, : 6 * HEAD_DIM].view(torch.uint8).cpu(),
            torch.zeros(NUM_HEADS, 6 * HEAD_DIM, dtype=torch.uint8),
        )
        expected_scale = bound_cpu.float() / FP8_MAX
        torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
        empty_history = torch.zeros(NUM_PAGES, dtype=torch.int32, device="npu")
        rescale_qfa_v_cache(self.values, self.scales, scale, empty_history, self.bound, self.head_scale)
        torch.testing.assert_close(self.bound.cpu(), bound_cpu, rtol=0, atol=0)
        torch.testing.assert_close(self.head_scale.cpu(), expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(self.raw.cpu(), self.initial, rtol=0, atol=0)

    def test_prepare_graph_replay_reads_changed_slot_validity(self):
        num_tokens = 6
        source = torch.linspace(-2, 2, num_tokens * NUM_HEADS * HEAD_DIM).reshape(num_tokens, NUM_HEADS, HEAD_DIM)
        source = source.to(torch.bfloat16)
        value = source.to("npu")
        slots = torch.arange(128, 128 + num_tokens, dtype=torch.int32, device="npu")
        bound = torch.tensor([0.5, 1.0], dtype=torch.bfloat16, device="npu")
        quantize_head_input(value, slots, bound)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            prepared, payload, scale = quantize_head_input(value, slots, bound)

        changed = source.clone()
        changed[1].fill_(1e20)
        changed[2].fill_(float("nan"))
        changed[4].fill_(-1e20)
        value.copy_(changed)
        slots.copy_(torch.tensor([128, -1, NUM_PAGES * BLOCK_SIZE, 129, -1, 130], dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        source[[1, 2, 4]] = 0
        expected = torch.cat(
            (source.permute(1, 0, 2).reshape(NUM_HEADS, -1), bound.cpu()[:, None].expand(-1, SENTINEL_SIZE)), dim=1
        )
        expected_payload, expected_scale = torch_npu.npu_dynamic_quant(
            expected.to("npu"), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        torch.testing.assert_close(prepared.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(payload.view(torch.uint8).cpu(), expected_payload.view(torch.uint8).cpu())
        torch.testing.assert_close(scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)

    def test_graph_replay_reads_changed_pages_lengths_and_head_scale(self):
        self.run_helpers()
        torch.npu.synchronize()
        self.raw.copy_(self.initial)
        head_scale_ptr = self.head_scale.data_ptr()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            codes = self.run_helpers()

        second_initial = self.initial.clone()
        values, scales = cache_views(second_initial)
        values[2].copy_(torch.full(values[2].shape, 2.0, dtype=torch.float32).to(torch.float8_e4m3fn))
        scales[2].copy_(torch.tensor([0.5, 2.0]))
        self.raw.copy_(second_initial)
        self.block_tables.copy_(torch.tensor([[2, 3, 0, -1], [0, 0, 0, -1], [0, 0, 0, -1]], dtype=torch.int32))
        self.cu_q.copy_(torch.tensor([0, 1, 1, 1], dtype=torch.int32))
        self.seq_lens.copy_(torch.tensor([129, 256, 256], dtype=torch.int32))
        self.new_scale.copy_(torch.tensor([2.0, 8.0]))
        graph.replay()
        torch.npu.synchronize()

        expected_codes = [0, 0, 129, 1]
        torch.testing.assert_close(codes.cpu(), torch.tensor(expected_codes, dtype=torch.int32))
        expected = expected_backing(second_initial, expected_codes, self.new_scale.cpu())
        torch.testing.assert_close(self.raw.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(self.bound.cpu(), (self.new_scale.cpu() * FP8_MAX).to(torch.bfloat16))
        torch.testing.assert_close(self.head_scale.cpu(), self.new_scale.cpu())
        self.assertEqual(self.head_scale.data_ptr(), head_scale_ptr)


if __name__ == "__main__":
    unittest.main()
