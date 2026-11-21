# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fixed normalized Sylvester H128 for post-RoPE C8 Q/K and learned sinks.

Seven butterfly stages accumulate in FP32, followed by one normalization
and one output cast. No random signs, calibration, input mutation or cache
state is involved. BF16 Q/K and an FP32 derived sink use the same basis.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

HADAMARD_DIM = 128
_ROWS_PER_PROGRAM = 16
_INV_SQRT_128 = 0.08838834764831845


@triton.jit
def _hadamard128_kernel(
    input_ptr,
    output_ptr,
    num_rows,
    HEADS: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    NORMALIZATION: tl.constexpr,
    ROWS: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    for tile in tl.range(tl.program_id(0), tl.cdiv(num_rows, ROWS), PROGRAMS):
        rows = tile * ROWS + tl.arange(0, ROWS)
        valid = rows < num_rows
        tokens = (rows // HEADS).to(tl.int64)
        heads = (rows % HEADS).to(tl.int64)
        offsets = tokens[:, None] * TOKEN_STRIDE + heads[:, None] * HEAD_STRIDE + cols[None, :] * DIM_STRIDE
        values = tl.load(input_ptr + offsets, valid[:, None], other=0).to(tl.float32)
        for stage in tl.static_range(7):
            bit = 1 << stage
            indices = tl.broadcast_to((cols ^ bit)[None, :], (ROWS, DIM))
            partner = tl.gather(values, indices, axis=1)
            values = tl.where((cols[None, :] & bit) == 0, values + partner, partner - values)
        values = values * NORMALIZATION
        tl.store(output_ptr + rows[:, None] * DIM + cols[None, :], values, valid[:, None])


def hadamard128(input: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Rotate strided [T,H,128] or [H,128], returning a fresh contiguous tensor.

    Production Q/K are BF16 and keep that dtype. The post-load sink requests
    FP32 output explicitly, retaining the loaded parameter without rounding
    the derived sink to BF16. This function has no CPU numerical fallback.
    """
    if input.ndim not in (2, 3) or input.shape[-1] != HADAMARD_DIM:
        raise ValueError("H128 requires [T,H,128] or [H,128].")
    output_dtype = input.dtype if output_dtype is None else output_dtype
    if input.dtype not in (torch.bfloat16, torch.float32) or output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("H128 supports BF16/FP32 input and output.")
    source = input.unsqueeze(0) if input.ndim == 2 else input
    # The Ascend gather lowering rejects a non-unit inner stride. Production
    # post-RoPE Q/K have unit inner stride; materialize only sliced channels.
    if source.stride(-1) != 1:
        source = source.contiguous()
    output = torch.empty(source.shape, dtype=output_dtype, device=input.device)
    rows = source.shape[0] * source.shape[1]
    if rows:
        init_device_properties_triton()
        programs = min(triton.cdiv(rows, _ROWS_PER_PROGRAM), get_vectorcore_num())
        _hadamard128_kernel[(programs,)](
            source,
            output,
            rows,
            HEADS=source.shape[1],
            TOKEN_STRIDE=source.stride(0),
            HEAD_STRIDE=source.stride(1),
            DIM_STRIDE=source.stride(2),
            DIM=HADAMARD_DIM,
            NORMALIZATION=_INV_SQRT_128,
            ROWS=_ROWS_PER_PROGRAM,
            PROGRAMS=programs,
        )
    return output.reshape(input.shape)
