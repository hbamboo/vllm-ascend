# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fused MXFP8 quantize + PA_NZ packet scatter for the C8 KV cache.

Replaces four kernels per layer (``npu_dynamic_mx_quant`` for K,
``npu_quantize`` for V, ``npu_scatter_pa_kv_cache`` and the Triton
``scatter_k_scales``) with a single kernel that

* computes the per-token-group E8M0 scale and FP8 E4M3 payload of K,
* converts V with the checkpoint's static per-channel reciprocal scale,
* writes K/V into the PA_NZ packet views and the K scales into the 6-D
  NZ scale view.

The K quantization follows ``npu_dynamic_mx_quant``: for every 32-element
group ``exp = floor(log2(amax)) - 8`` (all-zero group -> scale 1), payload
``= x / 2**exp`` rounded to E4M3 (RNE). The FP8 cast is the hardware
conversion (verified bit-identical to torch's ``float8_e4m3fn`` on NPU);
values above 448 are clamped first because the Triton cast yields NaN
where the CANN op saturates.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _quant_scatter_kernel(
    key_ptr,  # bf16 [T, N, D]
    value_ptr,  # bf16 [T, N, D]
    vscale_ptr,  # bf16 [N * D] reciprocal static scale
    slot_ptr,  # int32 [T]
    k_ptr,  # fp8 PA_NZ [Bn, N, D // 32, Bs, 32]
    v_ptr,  # fp8 PA_NZ [Bn, N, D // 32, Bs, 32]
    ks_ptr,  # e8m0 [Bn, N, Bs // 16, D // 64, 16, 2]
    num_pairs,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K_PAGE_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_PAGE_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    KS_PAGE_STRIDE: tl.constexpr,
    KS_HEAD_STRIDE: tl.constexpr,
    KS_SEG_STRIDE: tl.constexpr,
    KS_D1_STRIDE: tl.constexpr,
    KS_FRAG_STRIDE: tl.constexpr,
    KS_PAIR_STRIDE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in tl.static_range(TOKENS_PER_PROGRAM):
        pair_index = pid * TOKENS_PER_PROGRAM + i
        tok = pair_index // NUM_HEADS
        head = pair_index % NUM_HEADS
        if pair_index < num_pairs:
            slot = tl.load(slot_ptr + tok)
            if slot >= 0:
                blk = slot // BLOCK_SIZE
                off = slot % BLOCK_SIZE

                d32 = tl.arange(0, 32)
                row = tok * (NUM_HEADS * HEAD_DIM) + head * HEAD_DIM
                k_row = k_ptr + blk * K_PAGE_STRIDE + head * K_HEAD_STRIDE + off * 32
                v_row = v_ptr + blk * V_PAGE_STRIDE + head * V_HEAD_STRIDE + off * 32
                ks_base = (
                    blk * KS_PAGE_STRIDE
                    + head * KS_HEAD_STRIDE
                    + (off // 16) * KS_SEG_STRIDE
                    + (off % 16) * KS_FRAG_STRIDE
                )

                # K: four 32-element MX groups over the head dimension.
                for g in tl.static_range(HEAD_DIM // 32):
                    x = tl.load(key_ptr + row + g * 32 + d32).to(tl.float32)
                    amax = tl.max(tl.abs(x))
                    # floor(log2(amax)) via the fp32 exponent field, then -8.
                    abits = amax.to(tl.int32, bitcast=True)
                    e = ((abits >> 23) & 0xFF) - 135
                    e = tl.where(amax == 0.0, 0, e)
                    e = tl.minimum(tl.maximum(e, -127), 127)
                    # 2^e and 2^-e from the exponent field (no exp2/div).
                    scale = ((e + 127) << 23).to(tl.float32, bitcast=True)
                    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
                    qf = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
                    q = qf.to(tl.float8e4nv)
                    tl.store(k_row + g * (BLOCK_SIZE * 32) + d32, q)
                    sbyte = (e + 127).to(tl.uint8)
                    tl.store(
                        ks_ptr
                        + ks_base
                        + (g // 2) * KS_D1_STRIDE
                        + (g % 2) * KS_PAIR_STRIDE,
                        sbyte,
                    )

                # V: static per-channel scale, no per-token reduction.
                dv = tl.arange(0, HEAD_DIM)
                v = tl.load(value_ptr + row + dv).to(tl.float32)
                vs = tl.load(vscale_ptr + head * HEAD_DIM + dv).to(tl.float32)
                vf = tl.minimum(tl.maximum(v * vs, -448.0), 448.0)
                qv = vf.to(tl.float8e4nv)
                tl.store(v_row + (dv // 32) * (BLOCK_SIZE * 32) + (dv % 32), qv)


def quant_scatter_mxfp8(
    key: torch.Tensor,
    value: torch.Tensor,
    v_scale_reciprocal: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    tokens_per_program: int = 1,
    num_warps: int = 4,
) -> None:
    """Quantize K/V to MXFP8 and scatter into the C8 packet caches.

    ``key``/``value``: ``[T, N, D]`` bf16 (not yet quantized).
    ``key_cache``/``value_cache``: 5-D PA_NZ views ``(Bn, N, D // 32, Bs, 32)``
    whose dim-0 stride is the full packet pitch.
    ``k_scale_cache``: 6-D view ``(Bn, N, Bs // 16, D // 64, 16, 2)``.
    ``slot_mapping``: ``[T]`` int32; negative slots (padding) are skipped.
    """
    num_tokens, num_heads, head_dim = key.shape
    if num_tokens == 0:
        return
    block_size = key_cache.shape[3]
    num_pairs = num_tokens * num_heads
    grid = (triton.cdiv(num_pairs, tokens_per_program),)
    _quant_scatter_kernel[grid](
        key,
        value,
        v_scale_reciprocal,
        slot_mapping,
        key_cache,
        value_cache,
        k_scale_cache.view(torch.uint8) if k_scale_cache.dtype != torch.uint8 else k_scale_cache,
        num_pairs,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        K_PAGE_STRIDE=key_cache.stride(0),
        K_HEAD_STRIDE=key_cache.stride(1),
        V_PAGE_STRIDE=value_cache.stride(0),
        V_HEAD_STRIDE=value_cache.stride(1),
        KS_PAGE_STRIDE=k_scale_cache.stride(0),
        KS_HEAD_STRIDE=k_scale_cache.stride(1),
        KS_SEG_STRIDE=k_scale_cache.stride(2),
        KS_D1_STRIDE=k_scale_cache.stride(3),
        KS_FRAG_STRIDE=k_scale_cache.stride(4),
        KS_PAIR_STRIDE=k_scale_cache.stride(5),
        TOKENS_PER_PROGRAM=tokens_per_program,
        num_warps=num_warps,
    )
