# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Lazy V head-scale updates for native QFA FP8 paged attention."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

QFA_FP8_BLOCK_SIZE = 128
QFA_FP8_HEAD_DIM = 128
QFA_FP8_MAX = 448.0
QFA_V_HEAD_SENTINEL_SIZE = 32
_PREPARE_TOKEN_TILE = 32
_MARK_TILE = 128
_RESCALE_TOKEN_TILE = 32


@triton.jit
def _prepare_qfa_v_head_input_kernel(
    value,
    slot_mapping,
    head_bound,
    output,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    VALUE_DIM_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BOUND_STRIDE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_CACHE_SLOTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    SENTINEL_SIZE: tl.constexpr,
    NUM_TOKEN_TILES: tl.constexpr,
):
    head = tl.program_id(0)
    tile = tl.program_id(1)
    row_width = NUM_TOKENS * HEAD_DIM + SENTINEL_SIZE
    if tile < NUM_TOKEN_TILES:
        tokens = tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
        dimensions = tl.arange(0, HEAD_DIM)
        slots = tl.load(slot_mapping + tokens * SLOT_STRIDE, mask=tokens < NUM_TOKENS, other=-1)
        valid = (tokens < NUM_TOKENS) & (slots >= 0) & (slots < NUM_CACHE_SLOTS)
        values = tl.load(
            value
            + tokens[:, None] * VALUE_TOKEN_STRIDE
            + head * VALUE_HEAD_STRIDE
            + dimensions[None, :] * VALUE_DIM_STRIDE,
            mask=valid[:, None],
            other=0.0,
        )
        tl.store(
            output + head * row_width + tokens[:, None] * HEAD_DIM + dimensions[None, :],
            values,
            mask=tokens[:, None] < NUM_TOKENS,
        )
    else:
        tail = tl.arange(0, SENTINEL_SIZE)
        bound = tl.load(head_bound + head * BOUND_STRIDE)
        tl.store(output + head * row_width + NUM_TOKENS * HEAD_DIM + tail, bound)


def prepare_qfa_v_head_input(value, slot_mapping, head_bound, num_tokens, num_cache_slots):
    """Pack valid TND V into head rows, zero graph padding, and append bounds."""
    num_heads = value.shape[1]
    output = torch.empty(
        (num_heads, num_tokens * QFA_FP8_HEAD_DIM + QFA_V_HEAD_SENTINEL_SIZE),
        dtype=torch.bfloat16,
        device=value.device,
    )
    num_token_tiles = triton.cdiv(num_tokens, _PREPARE_TOKEN_TILE)
    init_device_properties_triton()
    _prepare_qfa_v_head_input_kernel[(num_heads, num_token_tiles + 1)](
        value,
        slot_mapping,
        head_bound,
        output,
        *value.stride(),
        slot_mapping.stride(0),
        head_bound.stride(0),
        num_tokens,
        num_cache_slots,
        QFA_FP8_HEAD_DIM,
        _PREPARE_TOKEN_TILE,
        QFA_V_HEAD_SENTINEL_SIZE,
        num_token_tiles,
    )
    return output


@triton.jit
def _qfa_v_history_pages_kernel(
    block_tables,
    cu_q,
    seq_lens,
    history_codes,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    CU_Q_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(0)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    seq_len = tl.load(seq_lens + request * SEQ_STRIDE)
    q_len = tl.load(cu_q + (request + 1) * CU_Q_STRIDE) - tl.load(cu_q + request * CU_Q_STRIDE)
    valid = (q_len > 0) & (columns < NUM_COLUMNS) & (columns * BLOCK_SIZE < seq_len)
    pages = tl.load(block_tables + request * TABLE_ROW_STRIDE + columns * TABLE_COL_STRIDE, mask=valid, other=-1)
    valid = valid & (pages >= 0) & (pages < NUM_PAGES)
    old_count = tl.minimum(tl.maximum(seq_len - q_len - columns * BLOCK_SIZE, 0), BLOCK_SIZE)
    # The extra one distinguishes referenced new pages from unreferenced pages.
    # Shared prefixes may appear in several rows; each page is rescaled once.
    tl.atomic_max(history_codes + pages, old_count + 1, mask=valid)


def qfa_v_history_pages(block_tables, cu_q, seq_lens, num_pages, block_size=QFA_FP8_BLOCK_SIZE):
    """Return per-page old-token counts plus one, or zero for unreferenced pages."""
    history_codes = torch.zeros(num_pages, dtype=torch.int32, device=block_tables.device)
    num_requests = min(block_tables.shape[0], cu_q.numel() - 1, seq_lens.numel())
    num_columns = block_tables.shape[1]
    if num_requests > 0 and num_columns > 0 and num_pages > 0:
        init_device_properties_triton()
        _qfa_v_history_pages_kernel[(num_requests, triton.cdiv(num_columns, _MARK_TILE))](
            block_tables,
            cu_q,
            seq_lens,
            history_codes,
            *block_tables.stride(),
            cu_q.stride(0),
            seq_lens.stride(0),
            num_columns,
            num_pages,
            block_size,
            _MARK_TILE,
        )
    return history_codes


@triton.jit
def _rescale_qfa_v_cache_kernel(
    value_cache,
    page_v_scales,
    new_head_scale,
    history_codes,
    head_bound,
    head_scale,
    VALUE_PAGE_STRIDE: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    VALUE_DIM_STRIDE: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    SCALE_HEAD_STRIDE: tl.constexpr,
    NEW_SCALE_STRIDE: tl.constexpr,
    BOUND_STRIDE: tl.constexpr,
    HEAD_SCALE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    page = tl.program_id(0)
    head = tl.program_id(1)
    new_scale = tl.load(new_head_scale + head * NEW_SCALE_STRIDE)
    if page == 0:
        tl.store(head_bound + head * BOUND_STRIDE, new_scale * FP8_MAX)
        tl.store(head_scale + head * HEAD_SCALE_STRIDE, new_scale)

    code = tl.load(history_codes + page)
    if code > 0:
        scale_ptr = page_v_scales + page * SCALE_PAGE_STRIDE + head * SCALE_HEAD_STRIDE
        old_count = code - 1
        # A fresh page may contain another hybrid cache type's former bytes.
        # Do not read its scale or payload before the caller scatters new V.
        if old_count > 0:
            old_scale = tl.load(scale_ptr)
            if old_scale != new_scale:
                ratio = old_scale / new_scale
                dimensions = tl.arange(0, HEAD_DIM)
                for token_start in range(0, BLOCK_SIZE, TOKEN_TILE):
                    tokens = token_start + tl.arange(0, TOKEN_TILE)
                    pointers = (
                        value_cache
                        + page * VALUE_PAGE_STRIDE
                        + head * VALUE_HEAD_STRIDE
                        + tokens[:, None] * VALUE_TOKEN_STRIDE
                        + dimensions[None, :] * VALUE_DIM_STRIDE
                    )
                    mask = tokens[:, None] < old_count
                    values = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
                    tl.store(pointers, (values * ratio).to(value_cache.dtype.element_ty), mask=mask)
        tl.store(scale_ptr, new_scale)


def rescale_qfa_v_cache(value_cache, page_v_scales, new_head_scale, history_codes, head_bound, head_scale):
    """Align referenced historical V to the new head scale before scattering new V.

    Unreferenced physical pages are never written, including pages currently
    owned by another hybrid cache type. Page/head pairs have unique writers.
    """
    num_pages, num_heads = value_cache.shape[:2]
    if num_pages > 0 and num_heads > 0:
        init_device_properties_triton()
        _rescale_qfa_v_cache_kernel[(num_pages, num_heads)](
            value_cache,
            page_v_scales,
            new_head_scale,
            history_codes,
            head_bound,
            head_scale,
            *value_cache.stride(),
            *page_v_scales.stride(),
            new_head_scale.stride(0),
            head_bound.stride(0),
            head_scale.stride(0),
            QFA_FP8_BLOCK_SIZE,
            QFA_FP8_HEAD_DIM,
            _RESCALE_TOKEN_TILE,
            QFA_FP8_MAX,
        )
