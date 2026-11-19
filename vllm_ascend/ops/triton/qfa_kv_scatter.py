# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bitwise BNBD K/V + FP32 K-scale scatter with 64-bit cache offsets."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_qfa_kv(
    key,
    value,
    scale,
    slots,
    key_cache,
    value_cache,
    scale_cache,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    KB: tl.constexpr,
    KN: tl.constexpr,
    KS: tl.constexpr,
    KC: tl.constexpr,
    VB: tl.constexpr,
    VN: tl.constexpr,
    VS: tl.constexpr,
    VC: tl.constexpr,
    SB: tl.constexpr,
    SN: tl.constexpr,
    SS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slots + token * SLOT_STRIDE).to(tl.int64)
    if slot >= 0:
        block = slot // BLOCK_TOKENS
        offset = slot % BLOCK_TOKENS
        d = tl.arange(0, HEAD_DIM).to(tl.int64)
        k = tl.load(key + token * KT + head * KH + d * KD)
        v = tl.load(value + token * VT + head * VH + d * VD)
        tl.store(key_cache + block * KB + head * KN + offset * KS + d * KC, k)
        tl.store(value_cache + block * VB + head * VN + offset * VS + d * VC, v)
        s = tl.load(scale + token * ST + head * SH)
        tl.store(scale_cache + block * SB + head * SN + offset * SS, s)


def scatter_qfa_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_scale: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter quantized bytes without requantization; skip negative slots.

    The delivered fused CANN scatter truncates K/V offsets above 4 GiB.
    Promote slot/head/channel indices before multiplying the packet strides.
    A zero head stride also supports the old broadcast per-token K scales.
    """
    tokens, heads, dim = key.shape
    if tokens == 0:
        return
    _scatter_qfa_kv[(tokens, heads)](
        key.view(torch.uint8),
        value.view(torch.uint8),
        key_scale,
        slot_mapping,
        key_cache.view(torch.uint8),
        value_cache.view(torch.uint8),
        scale_cache,
        *key.stride(),
        *value.stride(),
        *key_scale.stride(),
        slot_mapping.stride(0),
        *key_cache.stride(),
        *value_cache.stride(),
        *scale_cache.stride(),
        BLOCK_TOKENS=key_cache.shape[2],
        HEAD_DIM=dim,
        num_warps=4,
    )
