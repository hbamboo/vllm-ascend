#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Fused Triton kernel for the iQuest-Q1 learned sink-key correction.

The reference computation (see
``vllm_ascend/models/iquest_q1/attention.py::_apply_sink_key``) folds a
learned per-kv-head sink key into the attention output:

    logit[t, h]  = scale * sum_d(q[t, h, d] * sink_k[h // group, d])
    factor[t, h] = sigmoid(lse[t, h] - logit[t, h])
    out[t, h, d] *= factor[t, h]

The pure-torch implementation needs ~6 device kernels per call (bmm,
casts, Muls, Sub, Sigmoid, in-place Mul). At decode batch sizes every one
of those kernels costs a fixed few us of launch/tile overhead, which
dwarfs the actual data volume (~0.4MB per call per rank), so the chain
still costs ~2ms per decode step across 88 layers (2-2.5% of device time,
measured on 8x950DT at conc28).

This kernel fuses the whole chain into a single elementwise pass that
reads q, lse and sink directly (no permute copy) and updates the output
in place.

Numerics: bf16 products are exact in the fp32 accumulator, so versus the
torch path the only differences are (a) fp32 summation order inside the
dot product and (b) one ULP-level sigmoid difference - both far below the
bf16 output quantum. The kernel hard-codes the torch path's intermediate
bf16 roundings (sink cast, bmm output, factor cast) so the outputs stay
bit-identical for the vast majority of elements (microbench on 950DT:
100% bitwise over production-like and adversarial shapes).
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# Rows (one row = one (token, head) pair of HEAD_DIM elements) processed
# per program iteration. A (BLOCK_ROWS, HEAD_DIM) fp32 tile costs
# ~3 * BLOCK_ROWS * HEAD_DIM * 4 bytes of unified buffer, which keeps
# BLOCK_ROWS=32 at ~48KB - well within the 950DT UB budget.
_SINK_BLOCK_ROWS = 32


@triton.jit
def _triton_sink_apply(
    q_ptr,  # [T*H, D] bf16 (kv-major head layout, contiguous rows)
    out_ptr,  # [T*H, D] bf16, updated in place
    sink_ptr,  # [KV, D] fp32 master weights
    lse_ptr,  # [T*H] fp32 (TND LSE flattened)
    scale,
    num_rows,
    num_tokens,
    NUM_HEADS: tl.constexpr,  # H (per rank)
    GROUP: tl.constexpr,  # query heads per kv head
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    SINK_1D: tl.constexpr,  # single kv head: load the sink row once
    LSE_HEAD_MAJOR: tl.constexpr,  # lse is [H, T] (QFA) instead of [T*H]
    EXACT_FP32: tl.constexpr,  # C8 torch path: no intermediate bf16 roundings
):
    # One program per BLOCK_ROWS tile, no row loop: each tile has a serial
    # load -> dot -> sigmoid -> store dependency chain, so per-program
    # sequential tiles kill throughput on prefill (one program per tile keeps
    # every vector core busy instead). Max grid = 8192*H/32 = 1536 programs.
    pid = tl.program_id(0)
    d_offs = tl.arange(0, HEAD_DIM)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)

    mask = rows < num_rows
    if SINK_1D:
        # Single kv head per rank (this model under TP8): every row of the
        # tile shares one sink vector, so load it once as a flat 1-D load and
        # broadcast. The generic 2-D gather below has non-affine addresses
        # and defeats vectorized loads (~25x slower on prefill).
        sink_row = tl.load(sink_ptr + d_offs)
        if not EXACT_FP32:
            sink_row = sink_row.to(tl.bfloat16)
        sink_row = sink_row.to(tl.float32)
        sink = tl.broadcast_to(sink_row[None, :], (BLOCK_ROWS, HEAD_DIM))
    else:
        # head h belongs to kv head h // group (kv-major head layout)
        kv = (rows % NUM_HEADS) // GROUP
        sink = tl.load(
            sink_ptr + kv[:, None] * HEAD_DIM + d_offs[None, :],
            mask=mask[:, None],
            other=0,
        )
        if not EXACT_FP32:
            sink = sink.to(tl.bfloat16)
        sink = sink.to(tl.float32)
    q_tile = tl.load(
        q_ptr + rows[:, None] * HEAD_DIM + d_offs[None, :],
        mask=mask[:, None],
        other=0,
    ).to(tl.float32)
    logit = tl.sum(q_tile * sink, axis=1)
    if not EXACT_FP32:
        # replicate the FIA path's bf16 bmm output rounding
        logit = logit.to(tl.bfloat16).to(tl.float32)
    if LSE_HEAD_MAJOR:
        lse_off = (rows % NUM_HEADS) * num_tokens + rows // NUM_HEADS
    else:
        lse_off = rows
    lse_val = tl.load(lse_ptr + lse_off, mask=mask, other=0).to(tl.float32)
    factor = tl.sigmoid(lse_val - logit * scale)
    if not EXACT_FP32:
        # replicate factor.to(attn_output.dtype): >=0.998 rounds to exactly
        # 1.0 in bf16, keeping the common case bit-exact.
        factor = factor.to(tl.bfloat16).to(tl.float32)
    out_tile = tl.load(
        out_ptr + rows[:, None] * HEAD_DIM + d_offs[None, :],
        mask=mask[:, None],
        other=0,
    ).to(tl.float32)
    result = (out_tile * factor[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + rows[:, None] * HEAD_DIM + d_offs[None, :],
        result,
        mask=mask[:, None],
    )


def sink_apply_triton(
    query: torch.Tensor,
    attn_output: torch.Tensor,
    lse: torch.Tensor,
    sink_key: torch.Tensor,
    scale: float,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    lse_head_major: bool = False,
    exact_fp32: bool = False,
) -> None:
    """Fold the learned sink into ``attn_output[:num_tokens]`` in place.

    ``query`` and ``attn_output`` are the FIA operand/output slices with the
    kv-major head layout ``(num_tokens, num_heads, head_dim)`` (a flat
    ``(num_tokens, num_heads * head_dim)`` view works too); ``lse`` is the
    TND softmax LSE, ``(num_tokens, num_heads[, 1])``. ``attn_output`` must
    be contiguous (the FIA output buffer is) since the update is in place.
    """
    if not query.is_contiguous():
        query = query.contiguous()
    assert attn_output.is_contiguous(), "in-place sink update needs a contiguous output"
    q2d = query.reshape(num_tokens * num_heads, head_dim)
    out2d = attn_output.reshape(num_tokens * num_heads, head_dim)
    # LSE arrives as (T, H, 1) or (T, H); both flatten to one fp32 value per
    # (token, head) pair. Reshape first (free for contiguous input), then
    # materialize if a non-contiguous layout slips through.
    if lse_head_major:
        # QFA returns the LSE head-major, [num_heads, num_tokens]; the kernel
        # indexes it directly, so no transpose copy is needed.
        lse2d = lse.reshape(-1)
        assert lse2d.is_contiguous(), "head-major LSE must be contiguous"
    else:
        lse2d = lse.reshape(num_tokens, num_heads)
        if not lse2d.is_contiguous():
            lse2d = lse2d.contiguous()

    num_rows = num_tokens * num_heads
    if num_rows == 0:
        return
    init_device_properties_triton()  # idempotent; needed for standalone callers
    # One program per tile: max grid = 8192-token chunk -> 1536 programs, all
    # vector cores busy; smaller token counts launch proportionally less.
    grid = triton.cdiv(num_rows, _SINK_BLOCK_ROWS)
    _triton_sink_apply[(grid,)](
        q2d,
        out2d,
        sink_key,
        lse2d,
        scale,
        num_rows,
        num_tokens,
        NUM_HEADS=num_heads,
        GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_ROWS=_SINK_BLOCK_ROWS,
        SINK_1D=(num_kv_heads == 1),
        LSE_HEAD_MAJOR=lse_head_major,
        EXACT_FP32=exact_fp32,
    )


@triton.jit
def _triton_qfa_sink_apply(
    q_ptr,
    q_scale_ptr,
    attn_ptr,
    lse_ptr,
    sink_ptr,
    v_scale_ptr,
    out_ptr,
    scale,
    num_rows,
    NUM_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_SCALE_TOKEN_STRIDE: tl.constexpr,
    Q_SCALE_HEAD_STRIDE: tl.constexpr,
    LSE_HEAD_STRIDE: tl.constexpr,
    LSE_TOKEN_STRIDE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    SINK_1D: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
    V_SCALE_HEAD_STRIDE: tl.constexpr,
    V_SCALE_DIM_STRIDE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim = tl.arange(0, HEAD_DIM)
    valid = rows < num_rows
    token = rows // NUM_HEADS
    head = rows % NUM_HEADS
    if HAS_SINK:
        if Q_TOKEN_STRIDE == NUM_HEADS * HEAD_DIM and Q_HEAD_STRIDE == HEAD_DIM:
            q_offsets = rows[:, None] * HEAD_DIM + dim[None, :]
        else:
            q_offsets = token[:, None] * Q_TOKEN_STRIDE + head[:, None] * Q_HEAD_STRIDE + dim[None, :]
        query = tl.load(
            q_ptr + q_offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        query_scale = tl.load(
            q_scale_ptr + token * Q_SCALE_TOKEN_STRIDE + head * Q_SCALE_HEAD_STRIDE, mask=valid, other=0
        ).to(tl.float32)
        query = query * query_scale[:, None]
        if SINK_1D:
            sink_row = tl.load(sink_ptr + dim).to(tl.float32)
            sink = tl.broadcast_to(sink_row[None, :], (BLOCK_ROWS, HEAD_DIM))
        else:
            kv_head = head // GROUP
            sink = tl.load(sink_ptr + kv_head[:, None] * HEAD_DIM + dim[None, :], mask=valid[:, None], other=0).to(
                tl.float32
            )
        logit = tl.sum(query * sink, axis=1) * scale
        lse = tl.load(lse_ptr + head * LSE_HEAD_STRIDE + token * LSE_TOKEN_STRIDE, mask=valid, other=0).to(tl.float32)
        factor = tl.sigmoid(lse - logit)
    offsets = rows[:, None] * HEAD_DIM + dim[None, :]
    attn = tl.load(attn_ptr + offsets, mask=valid[:, None], other=0).to(tl.float32)
    if HAS_V_SCALE:
        if SINK_1D:
            # One KV head shares one channel vector across all query rows.
            # Use a vector load instead of the non-affine repeated gather.
            channel_row = tl.load(v_scale_ptr + dim * V_SCALE_DIM_STRIDE).to(tl.float32)
            channel_scale = tl.broadcast_to(channel_row[None, :], (BLOCK_ROWS, HEAD_DIM))
        else:
            channel_scale = tl.load(
                v_scale_ptr + (head // GROUP)[:, None] * V_SCALE_HEAD_STRIDE + dim[None, :] * V_SCALE_DIM_STRIDE,
                mask=valid[:, None],
                other=0,
            ).to(tl.float32)
        attn = attn * channel_scale
    if HAS_SINK:
        attn = attn * factor[:, None]
    tl.store(out_ptr + offsets, attn, mask=valid[:, None])


def qfa_sink_apply_triton(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    attn_output: torch.Tensor,
    lse: torch.Tensor | None,
    sink_key: torch.Tensor | None,
    scale: float,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    output: torch.Tensor,
    v_output_scale: torch.Tensor | None = None,
) -> None:
    """Write QFA output with the learned-key sink directly to ``output``.

    Q descales are FP32 ``[T]`` or ``[T, Hq]``. Optional static V scales
    ``[Hkv, D]`` are restored in the same kernel as the learned-key sink.
    Without a sink, only the V channel restoration is performed. Qhat,
    channel restoration and sink arithmetic stay FP32 until the final store.
    """
    num_rows = num_tokens * num_heads
    if num_rows == 0:
        return
    query = query.reshape(num_tokens, num_heads, head_dim)
    if sink_key is not None:
        lse = lse.reshape(num_heads, num_tokens)
    init_device_properties_triton()
    grid = triton.cdiv(num_rows, _SINK_BLOCK_ROWS)
    _triton_qfa_sink_apply[(grid,)](
        query,
        query_scale,
        attn_output,
        lse if sink_key is not None else attn_output,
        sink_key if sink_key is not None else query,
        v_output_scale if v_output_scale is not None else attn_output,
        output,
        scale,
        num_rows,
        NUM_HEADS=num_heads,
        GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        Q_TOKEN_STRIDE=query.stride(0),
        Q_HEAD_STRIDE=query.stride(1),
        Q_SCALE_TOKEN_STRIDE=query_scale.stride(0),
        Q_SCALE_HEAD_STRIDE=query_scale.stride(1) if query_scale.ndim == 2 else 0,
        LSE_HEAD_STRIDE=lse.stride(0) if sink_key is not None else 0,
        LSE_TOKEN_STRIDE=lse.stride(1) if sink_key is not None else 0,
        BLOCK_ROWS=_SINK_BLOCK_ROWS,
        SINK_1D=(num_kv_heads == 1),
        HAS_SINK=sink_key is not None,
        HAS_V_SCALE=v_output_scale is not None,
        V_SCALE_HEAD_STRIDE=v_output_scale.stride(0) if v_output_scale is not None else 0,
        V_SCALE_DIM_STRIDE=v_output_scale.stride(1) if v_output_scale is not None else 0,
    )
