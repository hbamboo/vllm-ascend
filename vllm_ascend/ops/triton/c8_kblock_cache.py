# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""FP8 K by token-block/channel and V by token/head for SWA.

Each physical page keeps two groups of original BF16 K for updates to an
unfinished group and speculative-token replacement. Quantization reads that
staging area before a separate launch updates it, so a large prefill cannot
overwrite K needed by another quantization program in the same page.
"""

import torch
from vllm.triton_utils import tl, triton

_FP8_MAX = 448.0
_GATHER_TOKEN_TILE = 32
_KEY_DIM_TILE = 32
_VALUE_TOKEN_TILE = 16


@triton.jit
def _divide_with_residual(numerator, denominator):
    # Ascend's reciprocal lowering can move an FP8 midpoint by one FP32
    # ULP. Correct both scale division and normalization before FP8 RNE.
    quotient = numerator / denominator
    residual = tl.fma(-quotient, denominator, numerator)
    return tl.fma(residual, 1.0 / denominator, quotient)


@triton.jit
def _quantize_kblock_keys(
    Key,
    KeyCache,
    KeyScale,
    KeyStage,
    Slots,
    QueryStarts,
    SeqLens,
    ActiveRequests,
    BlockTable,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_page_stride: tl.constexpr,
    key_scale_page_stride: tl.constexpr,
    stage_page_stride: tl.constexpr,
    table_stride: tl.constexpr,
    query_starts_size: tl.constexpr,
    seq_lens_size: tl.constexpr,
    table_rows: tl.constexpr,
    table_columns: tl.constexpr,
    num_tokens: tl.constexpr,
    slots_size: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DIM_TILE: tl.constexpr,
    DIM_TILES: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    request = tl.program_id(0)
    group_tile = tl.program_id(1)
    head = tl.program_id(2) // DIM_TILES
    dim_tile = tl.program_id(2) % DIM_TILES
    source_valid = request < tl.load(ActiveRequests)
    source_valid = source_valid & (request + 1 < query_starts_size) & (request < seq_lens_size) & (request < table_rows)
    q_start = tl.load(QueryStarts + request, source_valid, other=0)
    q_end = tl.load(QueryStarts + request + 1, source_valid, other=0)
    q_start = tl.minimum(tl.maximum(q_start, 0), num_tokens)
    q_end = tl.minimum(tl.maximum(q_end, 0), num_tokens)
    seq_len = tl.load(SeqLens + request, source_valid, other=0)
    computed = seq_len - (q_end - q_start)
    group_start = (computed // GROUP + group_tile) * GROUP
    active = source_valid & (q_end > q_start) & (computed >= 0) & (group_start < seq_len)
    active = active & (group_start // PAGE < table_columns)
    physical_page = tl.load(BlockTable + request * table_stride + group_start // PAGE, active, other=0)
    active = active & (physical_page > 0)
    first_new = tl.maximum(group_start, computed)
    first_token = q_start + first_new - computed
    first_slot = tl.load(Slots + first_token, active & (first_token < slots_size), other=-1)
    active = active & (first_slot == physical_page * PAGE + first_new % PAGE)

    row = tl.arange(0, GROUP)
    dim = dim_tile * DIM_TILE + tl.arange(0, DIM_TILE)
    positions = group_start + row
    current_tokens = q_start + positions - computed
    incoming = active & (positions >= computed) & (positions < seq_len) & (current_tokens < slots_size)
    slots = tl.load(Slots + current_tokens, incoming, other=-1)
    page_offsets = positions % PAGE
    incoming = incoming & (slots == physical_page * PAGE + page_offsets)
    valid = active & (positions < seq_len)
    history = valid & (positions < computed)
    old_rows = ((group_start % PAGE // GROUP) % 2) * GROUP + row
    old_key = tl.load(
        KeyStage
        + physical_page.to(tl.int64) * stage_page_stride
        + (old_rows[:, None] * HEADS + head) * DIM
        + dim[None, :],
        history[:, None] & (dim[None, :] < DIM),
        other=0,
    ).to(tl.float32)
    new_key = tl.load(
        Key + current_tokens[:, None] * key_token_stride + head * key_head_stride + dim[None, :],
        incoming[:, None] & (dim[None, :] < DIM),
        other=0,
    ).to(tl.float32)
    key = tl.where(incoming[:, None], new_key, old_key)
    maximum = tl.max(tl.abs(key), axis=0)
    key_scale = tl.where(maximum > 0, _divide_with_residual(maximum, FP8_MAX), 1.0)
    normalized_key = tl.minimum(tl.maximum(_divide_with_residual(key, key_scale[None, :]), -FP8_MAX), FP8_MAX)
    tl.store(
        KeyCache
        + physical_page.to(tl.int64) * key_page_stride
        + (page_offsets[:, None] * HEADS + head) * DIM
        + dim[None, :],
        normalized_key.to(KeyCache.dtype.element_ty, fp_downcast_rounding="rtne"),
        valid[:, None] & (dim[None, :] < DIM),
    )
    group_in_page = group_start % PAGE // GROUP
    tl.store(
        KeyScale + physical_page.to(tl.int64) * key_scale_page_stride + (group_in_page * HEADS + head) * DIM + dim,
        key_scale,
        active & (dim < DIM),
    )


@triton.jit
def _quantize_kblock_values(
    Value,
    ValueCache,
    ValueScale,
    Slots,
    QueryStarts,
    SeqLens,
    ActiveRequests,
    BlockTable,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_page_stride: tl.constexpr,
    value_scale_page_stride: tl.constexpr,
    table_stride: tl.constexpr,
    query_starts_size: tl.constexpr,
    seq_lens_size: tl.constexpr,
    table_rows: tl.constexpr,
    table_columns: tl.constexpr,
    num_tokens: tl.constexpr,
    slots_size: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    head = tl.program_id(2)
    source_valid = request < tl.load(ActiveRequests)
    source_valid = source_valid & (request + 1 < query_starts_size) & (request < seq_lens_size) & (request < table_rows)
    q_start = tl.load(QueryStarts + request, source_valid, other=0)
    q_end = tl.load(QueryStarts + request + 1, source_valid, other=0)
    q_start = tl.minimum(tl.maximum(q_start, 0), num_tokens)
    q_end = tl.minimum(tl.maximum(q_end, 0), num_tokens)
    seq_len = tl.load(SeqLens + request, source_valid, other=0)
    computed = seq_len - (q_end - q_start)
    current_tokens = q_start + tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    positions = computed + current_tokens - q_start
    valid = source_valid & (computed >= 0) & (current_tokens < q_end) & (current_tokens < slots_size)
    valid = valid & (positions // PAGE < table_columns)
    page = tl.load(BlockTable + request * table_stride + positions // PAGE, valid, other=0)
    valid = valid & (page > 0)
    slots = tl.load(Slots + current_tokens, valid, other=-1)
    valid = valid & (slots == page * PAGE + positions % PAGE)
    dim = tl.arange(0, PAD_DIM)
    value = tl.load(
        Value + current_tokens[:, None] * value_token_stride + head * value_head_stride + dim[None, :],
        valid[:, None] & (dim[None, :] < DIM),
        other=0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=1)
    scale = tl.where(maximum > 0, _divide_with_residual(maximum, FP8_MAX), 1.0)
    normalized = tl.minimum(tl.maximum(_divide_with_residual(value, scale[:, None]), -FP8_MAX), FP8_MAX)
    tl.store(
        ValueCache
        + page[:, None].to(tl.int64) * value_page_stride
        + ((positions[:, None] % PAGE) * HEADS + head) * DIM
        + dim[None, :],
        normalized.to(ValueCache.dtype.element_ty, fp_downcast_rounding="rtne"),
        valid[:, None] & (dim[None, :] < DIM),
    )
    tl.store(
        ValueScale + page.to(tl.int64) * value_scale_page_stride + (positions % PAGE) * HEADS + head,
        scale,
        valid,
    )


@triton.jit
def _update_kblock_stage(
    Key,
    KeyStage,
    Slots,
    QueryStarts,
    SeqLens,
    ActiveRequests,
    BlockTable,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    stage_page_stride: tl.constexpr,
    table_stride: tl.constexpr,
    query_starts_size: tl.constexpr,
    seq_lens_size: tl.constexpr,
    table_rows: tl.constexpr,
    table_columns: tl.constexpr,
    num_tokens: tl.constexpr,
    slots_size: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    DIM_TILE: tl.constexpr,
    DIM_TILES: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
):
    request = tl.program_id(0)
    group_tile = tl.program_id(1)
    head = tl.program_id(2) // DIM_TILES
    dim_tile = tl.program_id(2) % DIM_TILES
    source_valid = request < tl.load(ActiveRequests)
    source_valid = source_valid & (request + 1 < query_starts_size) & (request < seq_lens_size) & (request < table_rows)
    q_start = tl.load(QueryStarts + request, source_valid, other=0)
    q_end = tl.load(QueryStarts + request + 1, source_valid, other=0)
    q_start = tl.minimum(tl.maximum(q_start, 0), num_tokens)
    q_end = tl.minimum(tl.maximum(q_end, 0), num_tokens)
    seq_len = tl.load(SeqLens + request, source_valid, other=0)
    computed = seq_len - (q_end - q_start)
    group_start = (computed // GROUP + group_tile) * GROUP
    active = source_valid & (q_end > q_start) & (computed >= 0) & (group_start < seq_len)
    active = active & (group_start // PAGE < table_columns)
    physical_page = tl.load(BlockTable + request * table_stride + group_start // PAGE, active, other=0)
    page_last_position = tl.minimum(seq_len - 1, (group_start // PAGE + 1) * PAGE - 1)
    last_group = page_last_position % PAGE // GROUP
    group_in_page = group_start % PAGE // GROUP
    active = active & (physical_page > 0) & (group_in_page >= last_group - 1)
    row = tl.arange(0, GROUP)
    dim = dim_tile * DIM_TILE + tl.arange(0, DIM_TILE)
    positions = group_start + row
    current_tokens = q_start + positions - computed
    incoming = active & (positions >= computed) & (positions < seq_len) & (current_tokens < slots_size)
    slots = tl.load(Slots + current_tokens, incoming, other=-1)
    incoming = incoming & (slots == physical_page * PAGE + positions % PAGE)
    value = tl.load(
        Key + current_tokens[:, None] * key_token_stride + head * key_head_stride + dim[None, :],
        incoming[:, None] & (dim[None, :] < DIM),
        other=0,
    )
    stage_rows = (group_in_page % 2) * GROUP + row
    tl.store(
        KeyStage
        + physical_page.to(tl.int64) * stage_page_stride
        + (stage_rows[:, None] * HEADS + head) * DIM
        + dim[None, :],
        value,
        incoming[:, None] & (dim[None, :] < DIM),
    )


def update_kblock_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_reqs: int,
    active_num_reqs: torch.Tensor,
    group_size: int,
    max_query_len: int,
) -> None:
    """Update touched K groups from BF16 originals, then retain recent K.

    ``max_query_len`` is a host shape bound, not a device-derived scalar. It
    must also cover every request during a replay of the captured graph.
    Configuration validates page/group sizes and speculative rewind < G.
    """
    key_cache, value_cache, key_scale, value_scale, stage = kv_cache
    heads, dim = key.shape[1:]
    page = key_cache.shape[1]
    group_tiles = triton.cdiv(max_query_len + group_size - 1, group_size)
    dim_tiles = triton.cdiv(dim, _KEY_DIM_TILE)
    grid = (num_reqs, group_tiles, heads * dim_tiles)
    _quantize_kblock_keys[grid](
        key,
        key_cache,
        key_scale,
        stage,
        slot_mapping,
        query_start_loc,
        seq_lens,
        active_num_reqs,
        block_table,
        key.stride(0),
        key.stride(1),
        key_cache.stride(0),
        key_scale.stride(0),
        stage.stride(0),
        block_table.stride(0),
        query_start_loc.numel(),
        seq_lens.numel(),
        block_table.shape[0],
        block_table.shape[1],
        key.shape[0],
        slot_mapping.numel(),
        HEADS=heads,
        DIM=dim,
        DIM_TILE=_KEY_DIM_TILE,
        DIM_TILES=dim_tiles,
        PAGE=page,
        GROUP=group_size,
        FP8_MAX=_FP8_MAX,
    )
    _quantize_kblock_values[(num_reqs, triton.cdiv(max_query_len, _VALUE_TOKEN_TILE), heads)](
        value,
        value_cache,
        value_scale,
        slot_mapping,
        query_start_loc,
        seq_lens,
        active_num_reqs,
        block_table,
        value.stride(0),
        value.stride(1),
        value_cache.stride(0),
        value_scale.stride(0),
        block_table.stride(0),
        query_start_loc.numel(),
        seq_lens.numel(),
        block_table.shape[0],
        block_table.shape[1],
        value.shape[0],
        slot_mapping.numel(),
        HEADS=heads,
        DIM=dim,
        PAD_DIM=triton.next_power_of_2(dim),
        PAGE=page,
        TOKEN_TILE=_VALUE_TOKEN_TILE,
        FP8_MAX=_FP8_MAX,
    )
    _update_kblock_stage[grid](
        key,
        stage,
        slot_mapping,
        query_start_loc,
        seq_lens,
        active_num_reqs,
        block_table,
        key.stride(0),
        key.stride(1),
        stage.stride(0),
        block_table.stride(0),
        query_start_loc.numel(),
        seq_lens.numel(),
        block_table.shape[0],
        block_table.shape[1],
        key.shape[0],
        slot_mapping.numel(),
        HEADS=heads,
        DIM=dim,
        DIM_TILE=_KEY_DIM_TILE,
        DIM_TILES=dim_tiles,
        PAGE=page,
        GROUP=group_size,
    )


@triton.jit
def _gather_dequant_window(
    KeyCache,
    ValueCache,
    KeyScale,
    ValueScale,
    BlockTable,
    QueryStarts,
    SeqLens,
    ActiveRequests,
    WindowKey,
    WindowValue,
    key_page_stride: tl.constexpr,
    value_page_stride: tl.constexpr,
    key_scale_page_stride: tl.constexpr,
    value_scale_page_stride: tl.constexpr,
    table_stride: tl.constexpr,
    query_starts_size: tl.constexpr,
    seq_lens_size: tl.constexpr,
    table_rows: tl.constexpr,
    table_columns: tl.constexpr,
    num_tokens: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGES_PER_REQUEST: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    head = tl.program_id(2)
    source_valid = request < tl.load(ActiveRequests)
    source_valid = source_valid & (request + 1 < query_starts_size) & (request < seq_lens_size) & (request < table_rows)
    q_start = tl.load(QueryStarts + request, source_valid, other=0)
    q_end = tl.load(QueryStarts + request + 1, source_valid, other=0)
    q_start = tl.minimum(tl.maximum(q_start, 0), num_tokens)
    q_end = tl.minimum(tl.maximum(q_end, 0), num_tokens)
    seq_len = tl.load(SeqLens + request, source_valid, other=0)
    crop_len = tl.minimum(seq_len, WINDOW + q_end - q_start)
    row = tile * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    dim = tl.arange(0, PAD_DIM)
    position = seq_len - crop_len + row
    valid = source_valid & (row < crop_len) & (q_end > q_start) & (seq_len >= q_end - q_start)
    valid = valid & (position // PAGE < table_columns)
    page = tl.load(BlockTable + request * table_stride + position // PAGE, valid, other=0)
    valid = valid & (page > 0)
    page_offset = position % PAGE
    key = tl.load(
        KeyCache
        + page[:, None].to(tl.int64) * key_page_stride
        + (page_offset[:, None] * HEADS + head) * DIM
        + dim[None, :],
        valid[:, None] & (dim[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    key_scale = tl.load(
        KeyScale
        + page[:, None].to(tl.int64) * key_scale_page_stride
        + ((page_offset[:, None] // GROUP) * HEADS + head) * DIM
        + dim[None, :],
        valid[:, None] & (dim[None, :] < DIM),
        other=1,
    )
    value = tl.load(
        ValueCache
        + page[:, None].to(tl.int64) * value_page_stride
        + (page_offset[:, None] * HEADS + head) * DIM
        + dim[None, :],
        valid[:, None] & (dim[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    value_scale = tl.load(
        ValueScale + page.to(tl.int64) * value_scale_page_stride + page_offset * HEADS + head,
        valid,
        other=1,
    )
    output_offset = ((request.to(tl.int64) * PAGES_PER_REQUEST * PAGE + row[:, None]) * HEADS + head) * DIM + dim[
        None, :
    ]
    output_mask = (row[:, None] < PAGES_PER_REQUEST * PAGE) & (dim[None, :] < DIM)
    tl.store(WindowKey + output_offset, (key * key_scale).to(WindowKey.dtype.element_ty), output_mask)
    tl.store(WindowValue + output_offset, (value * value_scale[:, None]).to(WindowValue.dtype.element_ty), output_mask)


def gather_dequant_window(
    kv_cache: tuple[torch.Tensor, ...],
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    window_key: torch.Tensor,
    window_value: torch.Tensor,
    *,
    num_reqs: int,
    active_num_reqs: torch.Tensor,
    window_size: int,
    group_size: int,
    pages_per_request: int,
    num_tokens: int,
) -> None:
    """Read FP8 pages into a right-aligned, cropped BF16 FIA workspace.

    FIA's inclusive ``pre_tokens=W`` requires ``min(N, W+Q)`` keys. Padding
    after each request's cropped keys is overwritten with zero every run.
    """
    key_cache, value_cache, key_scale, value_scale, _ = kv_cache
    heads, dim = key_scale.shape[2:]
    page = key_cache.shape[1]
    _gather_dequant_window[(num_reqs, triton.cdiv(pages_per_request * page, _GATHER_TOKEN_TILE), heads)](
        key_cache,
        value_cache,
        key_scale,
        value_scale,
        block_table,
        query_start_loc,
        seq_lens,
        active_num_reqs,
        window_key,
        window_value,
        key_cache.stride(0),
        value_cache.stride(0),
        key_scale.stride(0),
        value_scale.stride(0),
        block_table.stride(0),
        query_start_loc.numel(),
        seq_lens.numel(),
        block_table.shape[0],
        block_table.shape[1],
        num_tokens,
        HEADS=heads,
        DIM=dim,
        PAD_DIM=triton.next_power_of_2(dim),
        PAGE=page,
        GROUP=group_size,
        WINDOW=window_size,
        PAGES_PER_REQUEST=pages_per_request,
        TOKEN_TILE=_GATHER_TOKEN_TILE,
    )
