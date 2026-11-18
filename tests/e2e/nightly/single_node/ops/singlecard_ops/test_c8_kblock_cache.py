# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real-NPU K-block/channel and V-token/head cache correctness checks.

Run directly from the repository root with ASCEND_RT_VISIBLE_DEVICES=0
and PYTHONPATH=. The bootstrap loads the production kernel file without
initializing vLLM's platform/plugin discovery.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu


def _cache(pages, heads, dim, group):
    page = 512
    shapes = (
        (pages, page, heads * dim),
        (pages, page, heads * dim),
        (pages, page // group, heads, dim),
        (pages, page, heads),
        (pages, 2 * group, heads, dim),
    )
    dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32, torch.float32, torch.bfloat16)
    sizes = [math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size() for shape, dtype in zip(shapes, dtypes)]
    pitch = sum(sizes) + 64
    raw = torch.full((pages * pitch,), 90, dtype=torch.int8, device="npu")
    caches, offset = [], 0
    for shape, dtype, size in zip(shapes, dtypes, sizes):
        element_size = torch.empty((), dtype=dtype).element_size()
        strides = (pitch // element_size, *(math.prod(shape[i + 1 :]) for i in range(1, len(shape))))
        caches.append(torch.as_strided(raw.view(dtype), shape, strides, offset // element_size))
        offset += size
    return raw, tuple(caches), pitch


def _quantized_reference(key, value, group):
    key = key.float()
    value = value.float()
    keys, key_scales = [], []
    for start in range(0, key.shape[0], group):
        tile = key[start : start + group]
        maximum = tile.abs().amax(dim=0)
        scale = torch.where(maximum > 0, maximum / 448.0, 1.0)
        keys.append((tile / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
        key_scales.append(scale.expand_as(tile))
    maximum = value.abs().amax(dim=-1, keepdim=True)
    value_scale = torch.where(maximum > 0, maximum / 448.0, 1.0)
    return (
        torch.cat(keys),
        torch.cat(key_scales),
        (value / value_scale).clamp(-448, 448).to(torch.float8_e4m3fn),
        value_scale,
    )


class TestKBlockChannelCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        import triton
        import triton.language as tl

        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        cls.kernel_path = root / "vllm_ascend/ops/triton/c8_kblock_cache.py"
        spec = importlib.util.spec_from_file_location("c8_kblock_cache_npu", cls.kernel_path)
        kernel = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl)}):
            spec.loader.exec_module(kernel)
        cls.update_kblock_cache = staticmethod(kernel.update_kblock_cache)
        cls.gather_dequant_window = staticmethod(kernel.gather_dequant_window)
        print("NPU", torch.npu.get_device_name(0), torch.__version__, torch_npu.__version__, flush=True)

    def test_active_dummy_without_a_slot(self):
        raw, cache, pitch = _cache(3, 1, 128, 32)
        key = torch.ones((3, 1, 128), dtype=torch.bfloat16, device="npu")
        starts = torch.tensor([0, 2, 3], dtype=torch.int32, device="npu")
        lengths = torch.tensor([2, 1], dtype=torch.int32, device="npu")
        active = torch.tensor([2], dtype=torch.int32, device="npu")
        table = torch.tensor([[2], [0]], dtype=torch.int32, device="npu")
        slots = torch.tensor([1024, 1025], dtype=torch.int64, device="npu")
        self.update_kblock_cache(
            key,
            key,
            cache,
            slots,
            starts,
            lengths,
            table,
            num_reqs=2,
            active_num_reqs=active,
            group_size=32,
            max_query_len=2,
        )
        out_k = torch.empty((2, 512, 128), dtype=torch.bfloat16, device="npu")
        out_v = torch.empty_like(out_k)
        self.gather_dequant_window(
            cache,
            table,
            starts,
            lengths,
            out_k,
            out_v,
            num_reqs=2,
            active_num_reqs=active,
            window_size=4096,
            group_size=32,
            pages_per_request=1,
            num_tokens=3,
        )
        self.assertEqual(torch.count_nonzero(out_k[1].cpu()).item(), 0)
        self.assertEqual(torch.count_nonzero(out_v[1].cpu()).item(), 0)
        self.assertTrue(bool((raw[:pitch].cpu() == 90).all()))

    def test_streaming_original_bf16_requantization(self):
        torch.manual_seed(41)
        heads, dim, capacity = 2, 128, 3
        # These include a many-group prefill, partial-group range expansion,
        # and speculative replacement across both group and page boundaries.
        steps = [
            (0, 73),
            (73, 6),
            (79, 532),
            (611, 6),
            (613, 6),
            (619, 14),
            (633, 6),
            (637, 6),
            (638, 6),
            (644, 373),
            (1017, 6),
            (1019, 6),
            (1020, 6),
        ]
        for group in (32, 64, 128):
            with self.subTest(group=group):
                raw, cache, pitch = _cache(6, heads, dim, group)
                table_cpu = torch.tensor([[2, 4, 1], [3, 5, 0], [5, 3, 0]], dtype=torch.int32)
                table = table_cpu.npu()
                active = torch.tensor([1], dtype=torch.int32, device="npu")
                history_k = torch.empty((0, heads, dim), dtype=torch.bfloat16)
                history_v = torch.empty_like(history_k)
                for computed, count in steps:
                    source_k = torch.randn((count, heads + 1, dim), dtype=torch.bfloat16)
                    source_v = torch.randn_like(source_k)
                    key, value = source_k[:, :heads], source_v[:, :heads]
                    # New extrema force old values in an unfinished K group
                    # to be requantized, with the original BF16 as the oracle.
                    key[-1, 0, 0] = 64 + computed / 8
                    value[:, 1] *= 8
                    history_k = torch.cat((history_k[:computed], key))
                    history_v = torch.cat((history_v[:computed], value))
                    total = computed + count
                    positions = torch.arange(computed, total)
                    slots = (table_cpu[0, positions // 512].long() * 512 + positions % 512).npu()
                    # Inactive rows deliberately contain plausible stale data;
                    # only the device active-request scalar identifies them.
                    starts = torch.tensor([0, count, 1, min(count, 5)], dtype=torch.int32, device="npu")
                    lengths = torch.tensor([total, 87, 95], dtype=torch.int32, device="npu")
                    self.update_kblock_cache(
                        key.npu(),
                        value.npu(),
                        cache,
                        slots,
                        starts,
                        lengths,
                        table,
                        num_reqs=capacity,
                        active_num_reqs=active,
                        group_size=group,
                        max_query_len=count,
                    )
                    k8, ks, v8, vs = _quantized_reference(history_k, history_v, group)
                    p = table_cpu[0, torch.arange(total) // 512].long()
                    r = torch.arange(total) % 512
                    got_k = cache[0].cpu().float()[p, r].reshape(total, heads, dim)
                    got_v = cache[1].cpu().float()[p, r].reshape(total, heads, dim)
                    got_ks = cache[2].cpu()[p, r // group]
                    got_vs = cache[3].cpu()[p, r].unsqueeze(-1)
                    torch.testing.assert_close(got_ks, ks, rtol=2e-7, atol=0)
                    torch.testing.assert_close(got_vs, vs, rtol=2e-7, atol=0)
                    torch.testing.assert_close(got_k.float(), k8.float(), rtol=0, atol=0)
                    torch.testing.assert_close(got_v.float(), v8.float(), rtol=0, atol=0)

                    window = 127
                    pages_per_request = math.ceil((window + count) / 512)
                    shape = (capacity * pages_per_request, 512, heads * dim)
                    out_k = torch.empty(shape, dtype=torch.bfloat16, device="npu")
                    out_v = torch.empty_like(out_k)
                    self.gather_dequant_window(
                        cache,
                        table,
                        starts,
                        lengths,
                        out_k,
                        out_v,
                        num_reqs=capacity,
                        active_num_reqs=active,
                        window_size=window,
                        group_size=group,
                        pages_per_request=pages_per_request,
                        num_tokens=count,
                    )
                    crop = min(total, window + count)
                    expected_k = (got_k.float() * got_ks).to(torch.bfloat16)[-crop:]
                    expected_v = (got_v.float() * got_vs).to(torch.bfloat16)[-crop:]
                    rows_k = out_k.cpu().reshape(capacity, -1, heads, dim)
                    rows_v = out_v.cpu().reshape(capacity, -1, heads, dim)
                    torch.testing.assert_close(rows_k[0, :crop], expected_k, rtol=0, atol=0)
                    torch.testing.assert_close(rows_v[0, :crop], expected_v, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(rows_k[0, crop:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(rows_v[0, crop:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(rows_k[1:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(rows_v[1:]).item(), 0)
                self.assertTrue(bool((raw[:pitch].cpu() == 90).all()))
                for inactive_page in (3, 5):
                    self.assertTrue(bool((raw[inactive_page * pitch : (inactive_page + 1) * pitch].cpu() == 90).all()))

    def test_zero_group_and_long_window(self):
        heads, dim, group, length = 1, 128, 32, 4231
        raw, cache, pitch = _cache(12, heads, dim, group)
        table_cpu = torch.tensor([[9, 3, 1, 5, 10, 2, 6, 4, 7]], dtype=torch.int32)
        table = table_cpu.npu()
        positions = torch.arange(length)
        slots = (table_cpu[0, positions // 512].long() * 512 + positions % 512).npu()
        starts = torch.tensor([0, length], dtype=torch.int32, device="npu")
        lengths = torch.tensor([length], dtype=torch.int32, device="npu")
        active = torch.ones((1,), dtype=torch.int32, device="npu")
        key = torch.zeros((length, heads, dim), dtype=torch.bfloat16, device="npu")
        value = torch.zeros_like(key)
        self.update_kblock_cache(
            key,
            value,
            cache,
            slots,
            starts,
            lengths,
            table,
            num_reqs=1,
            active_num_reqs=active,
            group_size=group,
            max_query_len=length,
        )
        p = table_cpu[0, positions // 512].long()
        r = positions % 512
        self.assertTrue(bool((cache[2].cpu()[p, r // group] == 1).all()))
        self.assertTrue(bool((cache[3].cpu()[p, r] == 1).all()))
        decode_starts = torch.tensor([0, 6], dtype=torch.int32, device="npu")
        pages_per_request = math.ceil((4096 + 6) / 512)
        out_k = torch.full((pages_per_request, 512, dim), 99, dtype=torch.bfloat16, device="npu")
        out_v = torch.empty_like(out_k)
        self.gather_dequant_window(
            cache,
            table,
            decode_starts,
            lengths,
            out_k,
            out_v,
            num_reqs=1,
            active_num_reqs=active,
            window_size=4096,
            group_size=group,
            pages_per_request=pages_per_request,
            num_tokens=6,
        )
        self.assertEqual(torch.count_nonzero(out_k.cpu()).item(), 0)
        self.assertEqual(torch.count_nonzero(out_v.cpu()).item(), 0)
        self.assertTrue(bool((raw[:pitch].cpu() == 90).all()))


if __name__ == "__main__":
    unittest.main()
