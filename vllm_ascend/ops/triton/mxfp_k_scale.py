# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Write packed K scales without copying back unrelated cache pages."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_elements"])
def _scatter_k_scale_kernel(
    source,
    cache,
    slot_mapping,
    num_elements,
    num_heads: tl.constexpr,
    num_groups: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    segment_stride: tl.constexpr,
    group_stride: tl.constexpr,
    fragment_stride: tl.constexpr,
    pair_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    block_tokens: tl.constexpr,
    token_fragment: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = index < num_elements
    token = index // (num_heads * num_groups * 2)
    head = index // (num_groups * 2) % num_heads
    group = index // 2 % num_groups
    pair = index % 2
    # Keep slot decomposition in this writer instead of launching six tensor
    # operations per cache group. Padding retains the reserved null-slot write.
    slot = tl.load(slot_mapping + token.to(tl.int64) * slot_stride, mask=valid, other=0).to(tl.int64)
    slot = tl.maximum(slot, 0)
    block = slot // block_tokens
    offset_in_block = slot % block_tokens
    segment = offset_in_block // token_fragment
    fragment = offset_in_block % token_fragment
    max_page_offset: tl.constexpr = (
        (num_heads - 1) * head_stride
        + (block_tokens // token_fragment - 1) * segment_stride
        + (num_groups - 1) * group_stride
        + (token_fragment - 1) * fragment_stride
        + pair_stride
    )
    # Normal packed pages fit INT32 internally. Keep the physical page base
    # INT64 without carrying six wide address vectors through the writer.
    # The constexpr branch also preserves views with larger internal strides.
    max_int32_offset: tl.constexpr = 2**31 - 1
    if max_page_offset <= max_int32_offset:
        page_offset = (
            head * head_stride
            + segment.to(tl.int32) * segment_stride
            + group * group_stride
            + fragment.to(tl.int32) * fragment_stride
            + pair * pair_stride
        )
    else:
        page_offset = (
            head.to(tl.int64) * head_stride
            + segment * segment_stride
            + group.to(tl.int64) * group_stride
            + fragment * fragment_stride
            + pair.to(tl.int64) * pair_stride
        )
    offset = block * block_stride + page_offset.to(tl.int64)
    value = tl.load(source + index, mask=valid, other=0)
    tl.store(cache + offset, value, mask=valid)


def scatter_k_scales(
    key_scale: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Update only the addressed token/head/group bytes of a PA_NZ view."""
    if slot_mapping.numel() == 0:
        return
    num_heads, num_groups = cache.shape[1], cache.shape[3]
    source = key_scale.view(torch.uint8).contiguous()
    assert source.numel() == slot_mapping.numel() * num_heads * num_groups * 2
    num_elements = source.numel()
    block_size = 256
    _scatter_k_scale_kernel[(triton.cdiv(num_elements, block_size),)](
        source,
        cache,
        slot_mapping,
        num_elements,
        num_heads,
        num_groups,
        *cache.stride(),
        slot_mapping.stride(0),
        cache.shape[2] * cache.shape[4],
        cache.shape[4],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
