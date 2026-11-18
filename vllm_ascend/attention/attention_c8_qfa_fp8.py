# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""QFA FP8 Q/K with dynamic FP32 token scales and checkpoint per-channel V."""

import torch
import torch_npu

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start


class AscendC8QFAFP8AttentionBackend(AscendC8MXFPAttentionBackend):
    @staticmethod
    def get_impl_cls():
        AscendC8MXFPAttentionBackend.get_impl_cls()
        return AscendC8QFAFP8AttentionBackendImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]


def scatter_qfa_fp8_cache(key, value, key_scale, kv_cache, slot_mapping):
    """Store payload bytes and FP32 K scales at matching PA_BNBD slots."""
    slots = slot_mapping.clamp(min=0).to(torch.int64)
    num_heads, block_size = kv_cache[0].shape[1:3]
    heads = torch.arange(num_heads, dtype=torch.int64, device=slots.device)
    indices = torch.stack(
        (
            (slots // block_size)[:, None].expand(-1, num_heads),
            heads[None, :].expand(slots.shape[0], -1),
            (slots % block_size)[:, None].expand(-1, num_heads),
        ),
        dim=-1,
    ).reshape(-1, 3)
    for payload, cache in zip((key, value), kv_cache[:2]):
        torch_npu.npu_scatter_nd_update_(
            cache.view(torch.int8), indices, payload.reshape(-1, key.shape[-1]).view(torch.int8)
        )
    scales = key_scale[:, None].expand(-1, num_heads).contiguous().reshape(-1)
    torch_npu.npu_scatter_nd_update_(kv_cache[2], indices, scales)
    notify_kv_cache_written()


class AscendC8QFAFP8AttentionBackendImpl(AscendC8MXFPAttentionBackendImpl):
    supports_quant_query_input = False

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QFA FP8 per-token does not support output quantization.")
        if attn_metadata is None:
            return output.zero_()
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output
        query = query[:num_tokens].reshape(num_tokens, self.num_heads, self.head_size)
        query8, query_scale = torch_npu.npu_dynamic_quant(
            query.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
        )
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key8, key_scale = torch_npu.npu_dynamic_quant(
                key[:num_tokens].reshape(num_tokens, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
            )
            value8 = torch_npu.npu_quantize(
                value[:num_tokens].reshape(num_tokens, -1),
                layer.v_cache_scale_float_reciprocal,
                None,
                torch.float8_e4m3fn,
                -1,
                False,
            )
            shape = (num_tokens, self.num_kv_heads, self.head_size)
            scatter_qfa_fp8_cache(
                key8.reshape(shape), value8.reshape(shape), key_scale, kv_cache, attn_metadata.slot_mapping[:num_tokens]
            )
        record_attention_compute_start()
        cu_q, used_kv = self._qfa_step_lengths(attn_metadata, num_tokens)
        max_q = attn_metadata.max_query_len or num_tokens
        mask_mode = 0 if max_q == 1 else 3
        plans = self._qfa_step_cache(attn_metadata)
        plan_key = ("fp8_token", self.num_heads, self.num_kv_heads, self.head_size, max_q, mask_mode)
        from cann_ops_transformer.ops import quant_flash_attn, quant_flash_attn_metadata

        metadata = plans.get(plan_key)
        if metadata is None:
            metadata = quant_flash_attn_metadata(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                6,
                cu_seqlens_q=cu_q,
                seqused_kv=used_kv,
                max_seqlen_q=max_q,
                max_seqlen_kv=-1,
                mask_mode=mask_mode,
                layout_q="NTD",
                layout_q_descale="NT",
                layout_kv="PA_BNBD",
                layout_out="TND",
            )
            plans[plan_key] = metadata
        query8 = query8.reshape_as(query)
        # Byte-view transpose avoids FP8 transpose falling back to AICPU.
        query_ntd = query8.view(torch.uint8).permute(1, 0, 2).contiguous().view(torch.float8_e4m3fn)
        scale_nt = query_scale[None, :].expand(self.num_heads, -1).contiguous()
        sink_key = getattr(layer, "sink_key", None)
        attn_output, lse = quant_flash_attn(
            query_ntd,
            kv_cache[0],
            kv_cache[1],
            scale_nt,
            kv_cache[2],
            self.qfa_v_descale,
            6,
            p_scale=self.qfa_p_scale,
            block_table=attn_metadata.block_tables,
            cu_seqlens_q=cu_q,
            seqused_kv=used_kv,
            metadata=metadata,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            attn_mask=self._qfa_int8_mask(attn_metadata) if mask_mode else None,
            max_seqlen_q=max_q,
            max_seqlen_kv=-1,
            layout_q="NTD",
            layout_q_descale="NT",
            layout_kv="PA_BNBD",
            layout_out="TND",
            return_softmax_lse=sink_key is not None,
        )
        owner = self
        if self.kv_sharing_target_layer_name is not None:
            owner = self.vllm_config.compilation_config.static_forward_context[self.kv_sharing_target_layer_name].impl
        group = self.num_heads // self.num_kv_heads
        channel_scale = owner.qfa_v_output_scale.repeat_interleave(group, dim=0)
        result = attn_output.float() * channel_scale[None, :, :]
        if sink_key is not None:
            # Use the same reconstructed Q for both QK logits and learned-key sink.
            q_hat = query8.float() * query_scale[:, None, None]
            q_hat = q_hat.reshape(num_tokens, self.num_kv_heads, group, self.head_size)
            sink_logits = (q_hat * sink_key.float().unsqueeze(1)).sum(-1).reshape(
                num_tokens, self.num_heads
            ) * self.scale
            lse = lse.reshape(self.num_heads, num_tokens).t()
            result *= torch.sigmoid(lse.float() - sink_logits)[..., None]
        output[:num_tokens] = result.to(output.dtype).reshape_as(output[:num_tokens])
        return output
