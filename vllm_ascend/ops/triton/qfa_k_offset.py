# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Causal online K offsets stored in each physical QFA cache packet."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _initialize_anchor(
    key, slots, anchor, descale,
    KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    AB: tl.constexpr, AH: tl.constexpr, AD: tl.constexpr,
    SB: tl.constexpr, SH: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slots + token * SLOT_STRIDE).to(tl.int64)
    # Exactly one actual token opens a page. Later append/prefix steps retain
    # this immutable anchor; physical-page reassignment writes a new first key.
    if slot >= 0 and slot % BLOCK_TOKENS == 0:
        block = slot // BLOCK_TOKENS
        d = tl.arange(0, HEAD_DIM).to(tl.int64)
        k = tl.load(key + token * KT + head * KH + d * KD).to(tl.float32)
        scale = tl.max(tl.abs(k), 0) / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)
        code = tl.minimum(tl.maximum(k / scale, -448.0), 448.0)
        tl.store(
            anchor + block * AB + head * AH + d * AD,
            code.to(anchor.dtype.element_ty, fp_downcast_rounding="rtne"),
        )
        tl.store(descale + block * SB + head * SH, scale)


@triton.jit
def _quantize_offset_k(
    key, slots, anchor, anchor_descale, codes, scales,
    KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    AB: tl.constexpr, AH: tl.constexpr, AD: tl.constexpr,
    SB: tl.constexpr, SH: tl.constexpr,
    NUM_HEADS: tl.constexpr, BLOCK_TOKENS: tl.constexpr, HEAD_DIM: tl.constexpr,
    SCALAR: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, HEAD_DIM).to(tl.int64)
    slot = tl.load(slots + token * SLOT_STRIDE).to(tl.int64)
    code_base = (token * NUM_HEADS + head) * HEAD_DIM
    if slot >= 0:
        block = slot // BLOCK_TOKENS
        k = tl.load(key + token * KT + head * KH + d * KD).to(tl.float32)
        if SCALAR:
            offset = (tl.max(k, 0) + tl.min(k, 0)) * 0.5
            tl.store(anchor + block * AB + head * AH + (slot % BLOCK_TOKENS) * AD, offset)
        else:
            a = tl.load(anchor + block * AB + head * AH + d * AD).to(tl.float32)
            s = tl.load(anchor_descale + block * SB + head * SH)
            offset = a * s
        # FP32 subtraction is deliberate: rounding this residual to BF16
        # before quantizing is a different control and loses the CPU benefit.
        residual = k - offset
        scale = tl.max(tl.abs(residual), 0) / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)
        code = tl.minimum(tl.maximum(residual / scale, -448.0), 448.0)
        tl.store(codes + code_base + d, code.to(codes.dtype.element_ty, fp_downcast_rounding="rtne"))
        tl.store(scales + token * NUM_HEADS + head, scale)
    else:
        # Graph padding must not read or mutate a real cache page.
        tl.store(codes + code_base + d, 0.0)
        tl.store(scales + token * NUM_HEADS + head, 1.0)


def quantize_qfa_k_offset(key, slot_mapping, kv_cache, mode):
    """Quantize residual K and write its online offset; no D2H synchronization.

    Initialization and residual quantization are separate ordered launches.
    This supports blocks opening in the middle of a prefill chunk without
    relying on execution order between Triton programs within one launch.
    """
    tokens, heads, dim = key.shape
    codes = torch.empty(key.shape, dtype=torch.float8_e4m3fn, device=key.device)
    scales = torch.empty((tokens, heads), dtype=torch.float32, device=key.device)
    if tokens == 0:
        return codes, scales
    offset = kv_cache[4]
    scalar = mode == "per_token_head_scalar"
    descale = kv_cache[5] if not scalar else scales
    descale_strides = descale.stride() if not scalar else (0, 0)
    common = (
        *key.stride(), slot_mapping.stride(0), *offset.stride(), *descale_strides,
    )
    if not scalar:
        _initialize_anchor[(tokens, heads)](
            key, slot_mapping, offset, descale, *common,
            BLOCK_TOKENS=kv_cache[0].shape[2], HEAD_DIM=dim,
            num_warps=4, enable_fp_fusion=False,
        )
    _quantize_offset_k[(tokens, heads)](
        key, slot_mapping, offset, descale, codes, scales, *common,
        NUM_HEADS=heads, BLOCK_TOKENS=kv_cache[0].shape[2], HEAD_DIM=dim,
        SCALAR=scalar, num_warps=4, enable_fp_fusion=False,
    )
    return codes, scales
