# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_aicore_num

from .utils import prepare_chunk_indices, safe_exp


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
# bh_step/task_num 随 batch 形状(序列数 × chunk 数)变化, 必须是运行期参数:
# 作为 tl.constexpr 时每个新值都是一条新的 cache key, 现场重编译一次
# (ttir→npubin ~2.5 s, 阻塞整个前向 → 该 step 的 KV 迟迟不出, TTFT 劣化).
# 二者只用于 tl.range 边界与整除, 运行期合法, 与 T/B 同样 do_not_specialize.
@triton.jit(do_not_specialize=["T", "B", "task_num", "bh_step"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,  # [H, B, T]
    g_cumsum,  # [H, B, T]
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    bh_step,
    task_num,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    num_core: tl.constexpr,
):
    bt_stride = B * T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_i = task_id // bh_step
        i_bh = task_id % bh_step
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_i * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_i * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_t_i
        o_t = tl.arange(0, BT)
        o_t_fp32 = o_t.to(tl.float32)

        p_beta = tl.make_block_ptr(beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * Hg + i_h // (H // Hg)) * K, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))

        if USE_G:
            p_g = tl.make_block_ptr(g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A *= safe_exp(b_g_diff)

        b_A *= b_beta[:, None]
        b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
        p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, Hg, K = k.shape

    H = beta.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    num_core = get_aicore_num()
    bh_step = B * H
    task_num = NT * bh_step

    from vllm_ascend.device.device_op import DeviceOperator

    A = DeviceOperator.chunk_scaled_dot_kkt_fwd(
        num_core=num_core,
        bh_step=bh_step,
        task_num=task_num,
        k=k,
        beta=torch.permute(beta, (2, 0, 1)).contiguous(),
        g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous(),
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=128,
    )
    return A
