# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_ascend.attention.attention_c8_mxfp import (
    fia_kblock_channel_packet_section_sizes,
    get_c8_swa_k_block_size,
    mxfp_cache_spec,
    mxfp_cache_views_for_spec,
)
from vllm_ascend.worker.utils import AscendKVBlockZeroer, copy_kv_cache_blocks_inplace


def _spec(group_size, ratio=1, heads=1, padding=0):
    spec = mxfp_cache_spec(
        SlidingWindowSpec(
            block_size=512 * ratio,
            num_kv_heads=heads,
            head_size=128,
            dtype=torch.float8_e4m3fn,
            sliding_window=4096,
            extra_retained_tokens=5,
        ),
        fia_pseudo=True,
        k_block_size=group_size,
    )
    return replace(spec, page_size_padded=spec.page_size_bytes + padding * ratio)


@pytest.mark.parametrize(
    "group_size,state_bytes,kernel_bytes", [(32, 308, 157696), (64, 332, 169984), (128, 392, 200704)]
)
def test_spec_budgets_staging_for_every_kernel_block(group_size, state_bytes, kernel_bytes):
    spec = _spec(group_size, ratio=2)
    assert spec.state_content_bytes == state_bytes
    assert spec.page_size_bytes == 2 * kernel_bytes
    assert (spec.sliding_window, spec.extra_retained_tokens) == (4096, 5)
    assert spec.head_size == spec.head_size_v == 128
    # A FullAttentionSpec may still belong to a SWA implementation after
    # hybrid cache grouping promotes its storage policy.
    promoted = mxfp_cache_spec(
        FullAttentionSpec(block_size=1024, num_kv_heads=1, head_size=128, dtype=torch.float8_e4m3fn),
        fia_pseudo=True,
        k_block_size=group_size,
    )
    assert promoted.state_content_bytes == state_bytes
    assert promoted.page_size_bytes == spec.page_size_bytes


@pytest.mark.parametrize("group_size,ratio,heads", [(32, 2, 2), (64, 1, 1), (128, 2, 1)])
def test_five_views_partition_pages_and_preserve_padding_and_guards(group_size, ratio, heads):
    spec = _spec(group_size, ratio=ratio, heads=heads, padding=256)
    offset, guard, pages = 128, 128, 2
    backing = torch.full((offset + pages * spec.page_size_bytes + guard,), 89, dtype=torch.int8)
    raw = backing[offset:-guard]
    views = mxfp_cache_views_for_spec(raw, spec, 512, fia_pseudo=True, k_block_size=group_size)
    blocks, pitch = pages * ratio, spec.page_size_bytes // ratio
    sizes = (
        512 * heads * 128,
        512 * heads * 128,
        512 // group_size * heads * 128 * 4,
        512 * heads * 4,
        2 * group_size * heads * 128 * 2,
    )
    shapes = (
        (blocks, 512, heads * 128),
        (blocks, 512, heads * 128),
        (blocks, 512 // group_size, heads, 128),
        (blocks, 512, heads),
        (blocks, 2 * group_size, heads, 128),
    )
    dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32, torch.float32, torch.bfloat16)
    start = 0
    for marker, (view, shape, dtype, size) in enumerate(zip(views, shapes, dtypes, sizes), start=1):
        assert view.shape == shape and view.dtype == dtype
        assert view.stride(0) * view.element_size() == pitch
        assert view.storage_offset() * view.element_size() == offset + start
        assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert view[0].is_contiguous()
        view.view(torch.uint8).fill_(marker)
        start += size
    physical_pages = raw.view(blocks, pitch)
    start = 0
    for marker, size in enumerate(sizes, start=1):
        assert (physical_pages[:, start : start + size] == marker).all()
        start += size
    assert (physical_pages[:, start:] == 89).all()
    assert (backing[:offset] == 89).all() and (backing[-guard:] == 89).all()


def test_copy_and_zero_include_bf16_staging_without_touching_padding():
    ratio, group_size = 2, 32
    spec = _spec(group_size, ratio=ratio, padding=256)
    offset, guard, pages = 128, 128, 3
    backing = torch.full((offset + pages * spec.page_size_bytes + guard,), 91, dtype=torch.int8)
    raw = backing[offset:-guard]
    views = mxfp_cache_views_for_spec(raw, spec, 512, fia_pseudo=True, k_block_size=group_size)
    for section, view in enumerate(views):
        for block in range(pages * ratio):
            view[block].view(torch.uint8).fill_(section * 17 + block + 1)
    snapshots = [view.view(torch.uint8).clone().unflatten(0, (pages, ratio)) for view in views]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2), KVCacheBlockCopy(src_block_id=2, dst_block_id=0)]
    copy_kv_cache_blocks_inplace([views], pages, copies)
    for view, snapshot in zip(views, snapshots):
        data = view.view(torch.uint8).unflatten(0, (pages, ratio))
        assert torch.equal(data[0], snapshot[2])
        assert torch.equal(data[2], snapshot[0])
        assert torch.equal(data[1], snapshot[1])
    before = backing.clone()
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["swa"])
    zeroer.init_meta([group], [512], "mxfp8", set(), {"swa": SimpleNamespace(kv_cache=views)})
    addresses, sizes, _, _, count = zeroer._meta
    assert count == 5 * ratio
    for address, size, stride in zip(addresses.tolist(), sizes.tolist(), zeroer._seg_page_strides.tolist()):
        start = address - raw.data_ptr() + 4 * stride
        raw[start : start + size * 4].zero_()
    for view in views:
        assert (view[ratio : 2 * ratio].view(torch.uint8) == 0).all()
    assert torch.equal(backing[: offset + spec.page_size_bytes], before[: offset + spec.page_size_bytes])
    assert torch.equal(backing[offset + 2 * spec.page_size_bytes :], before[offset + 2 * spec.page_size_bytes :])
    pitch = spec.page_size_bytes // ratio
    assert (raw.view(pages * ratio, pitch)[:, 157696:] == 91).all()


def test_layer_selection_uses_impl_window_and_defaults_to_original_layout():
    swa = SimpleNamespace(impl=SimpleNamespace(sliding_window=4096))
    full = SimpleNamespace(impl=SimpleNamespace(sliding_window=None))
    assert get_c8_swa_k_block_size(swa, 32) == 32
    assert get_c8_swa_k_block_size(full, 32) == 0
    assert get_c8_swa_k_block_size(swa, 0) == 0
    assert get_c8_swa_k_block_size(None, 32) == 0
    spec = _spec(32)
    original = mxfp_cache_spec(replace(spec, state_content_bytes=None), fia_pseudo=True)
    assert original.state_content_bytes == 264
    raw = torch.empty(original.page_size_bytes, dtype=torch.int8)
    assert len(mxfp_cache_views_for_spec(raw, original, 512, fia_pseudo=True)) == 4


@pytest.mark.parametrize("block_size,group_size", [(32, 32), (512, 48)])
def test_invalid_group_geometry_is_rejected(block_size, group_size):
    with pytest.raises(ValueError, match="whole groups and at least two"):
        fia_kblock_channel_packet_section_sizes(1, block_size, 128, group_size)
