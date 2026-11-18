# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""BF16 Q and dynamic per-token FP8 K/V, dequantized inside FIA v2."""

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.attention.attention_v1 import _FIA_V2_WORKSPACE_KEY, SWA_INT_MAX, FIAV2ParamProvider
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.sink import sink_apply_triton


class AscendC8FIAPseudoAttentionBackend(AscendC8MXFPAttentionBackend):
    @staticmethod
    def get_impl_cls():
        AscendC8MXFPAttentionBackend.get_impl_cls()
        return AscendC8FIAPseudoAttentionBackendImpl


def scatter_fia_fp8_cache(key, value, key_scale, value_scale, kv_cache, slot_mapping):
    """Write FP8 K/V and FP32 scales to the same physical token slots."""
    # Block 0 is vLLM's reserved null block for padded rows.
    slots = slot_mapping.clamp(min=0).to(torch.int64)
    block_size = kv_cache[0].shape[1]
    indices = torch.stack((slots // block_size, slots % block_size), dim=-1)
    for payload, cache in zip((key, value), kv_cache[:2]):
        updates = payload.reshape(payload.shape[0], -1).view(torch.int8)
        torch_npu.npu_scatter_nd_update_(cache.view(torch.int8), indices, updates)
    for payload, cache in zip((key_scale, value_scale), kv_cache[2:]):
        torch_npu.npu_scatter_nd_update_(cache, indices, payload.reshape(-1))
    notify_kv_cache_written()


class AscendC8FIAPseudoAttentionBackendImpl(AscendC8MXFPAttentionBackendImpl):
    supports_quant_query_input = False
    # create_weights installs this implementation with __class__ assignment,
    # so no constructor runs. The optional tensors are populated after load.
    qk_smoothing_query_scale: torch.Tensor | None = None
    qk_smoothing_key_scale: torch.Tensor | None = None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("FIA per-token KV does not support output quantization.")
        if attn_metadata is None:
            return output.zero_()
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens].contiguous()
        if num_tokens == 0:
            return output
        attention_query = query
        if self.qk_smoothing_query_scale is not None:
            attention_query = (query.float() * self.qk_smoothing_query_scale).to(query.dtype)
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key = key[:num_tokens]
            if self.qk_smoothing_key_scale is not None:
                key = (key.float() * self.qk_smoothing_key_scale).to(key.dtype)
            key8, key_scale = torch_npu.npu_dynamic_quant(
                key.reshape(num_tokens, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
            )
            value8, value_scale = torch_npu.npu_dynamic_quant(
                value[:num_tokens].reshape(num_tokens, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
            )
            kv_shape = (num_tokens, self.num_kv_heads, self.head_size)
            scatter_fia_fp8_cache(
                key8.view(kv_shape),
                value8.view(kv_shape),
                key_scale,
                value_scale,
                kv_cache,
                attn_metadata.slot_mapping[:num_tokens],
            )
        record_attention_compute_start()
        # CANN's single-query FP8 path only accepts dense attention. Prepend
        # one temporary query to each SWA singleton so the banded kernel is
        # selected. The last row keeps its original absolute KV position;
        # no dummy KV token is written to the cache.
        ends = attn_metadata.actual_seq_lengths_q
        pad_single_query = self.sliding_window is not None and any(
            end - start == 1 for start, end in zip([0, *ends[:-1]], ends)
        )
        fia_query = attention_query
        real_slices = []
        if pad_single_query:
            chunks, padded_ends = [], []
            padded_start = 0
            for start, end in zip([0, *ends[:-1]], ends):
                singleton = end - start == 1
                chunk = attention_query[start:end]
                if singleton:
                    chunk = chunk.expand(2, -1, -1)
                chunks.append(chunk)
                padded_end = padded_start + chunk.shape[0]
                real_slices.append(slice(padded_start + singleton, padded_end))
                padded_ends.append(padded_end)
                padded_start = padded_end
            fia_query = torch.cat(chunks)
        kwargs = self._fia_kwargs(fia_query, kv_cache, attn_metadata)
        if pad_single_query:
            kwargs["actual_seq_qlen"] = padded_ends
        if _EXTRA_CTX.capturing:
            workspace = get_capture_resource(
                (_FIA_V2_WORKSPACE_KEY, "fp8_per_token", self.sliding_window),
                lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kwargs),
                self._use_max_workspace_for_fia_graph,
            )
            lse = torch.empty((fia_query.shape[0], self.num_heads, 1), dtype=torch.float32, device=query.device)
            attn_output = (
                torch.empty_like(fia_query)
                if pad_single_query
                else output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
            )
            register_task(
                torch_npu.npu_fused_infer_attention_score_v2.out,
                {**kwargs, "workspace": workspace, "out": [attn_output, lse]},
                FIAV2ParamProvider(self._graph_metadata_layer_name(layer), pad_single_query=pad_single_query),
            )
        else:
            attn_output, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
        if pad_single_query:
            attn_output = torch.cat([attn_output[rows] for rows in real_slices])
            lse = torch.cat([lse[rows] for rows in real_slices])
        sink_key = getattr(layer, "sink_key", None)
        if sink_key is not None:
            sink_apply_triton(
                query,
                attn_output,
                lse,
                sink_key,
                self.scale,
                num_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
            )
        if not _EXTRA_CTX.capturing or pad_single_query:
            output[:num_tokens] = attn_output
        return output

    def _fia_kwargs(self, query, kv_cache, metadata):
        ends = metadata.actual_seq_lengths_q
        causal = metadata.causal and any(end - start > 1 for start, end in zip([0, *ends[:-1]], ends))
        windowed = self.sliding_window is not None
        return {
            "query": query,
            "key": kv_cache[0],
            "value": kv_cache[1],
            "dequant_scale_key": kv_cache[2],
            "dequant_scale_value": kv_cache[3],
            "query_quant_mode": 0,
            "key_quant_mode": 4,
            "value_quant_mode": 4,
            "input_layout": "TND",
            "num_query_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "softmax_scale": self.scale,
            "block_table": metadata.block_tables,
            "block_size": kv_cache[0].shape[1],
            "actual_seq_qlen": ends,
            "actual_seq_kvlen": metadata.seq_lens_list,
            "atten_mask": metadata.attn_mask if windowed or causal else None,
            "sparse_mode": 4 if windowed else 3 if causal else 0,
            "pre_tokens": self.sliding_window if windowed else SWA_INT_MAX,
            "next_tokens": 0 if windowed or causal else SWA_INT_MAX,
            "return_softmax_lse": True,
        }
