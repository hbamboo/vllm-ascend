# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""BF16 Q + dynamic per-token FP8 KV, dequantized inside FIA v2.

K and V each have an FP32 scale per token, stored alongside their physical
cache page. The checkpoint's static V scale is unused by this backend.
"""

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.attention.attention_v1 import (
    _FIA_V2_WORKSPACE_KEY,
    SWA_INT_MAX,
    FIAV2ParamProvider,
)
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.device.device_config import is_950
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.sink import sink_apply_triton


class AscendC8FIAPseudoAttentionBackend(AscendC8MXFPAttentionBackend):
    @staticmethod
    def get_impl_cls():
        # Retain the existing PCP/DCP capability checks and metadata builder.
        AscendC8MXFPAttentionBackend.get_impl_cls()
        return AscendC8FIAPseudoAttentionBackendImpl


def scatter_fia_fp8_cache(key, value, key_scale, value_scale, kv_cache, slot_mapping):
    """Update FP8 K/V and FP32 scales at the same physical page and offset."""
    if slot_mapping.numel() == 0:
        return
    # vLLM reserves block 0 as its null block. Dummy/padded rows write there
    # rather than interpreting PAD_SLOT_ID=-1 as the last live cache row.
    slots = slot_mapping.clamp(min=0).to(torch.int64)
    block_size = kv_cache[0].shape[1]
    indices = torch.stack((slots // block_size, slots % block_size), dim=-1)
    for payload, cache in zip((key, value), kv_cache[:2]):
        # Keep the page stride: flattening the cache would copy packet views.
        updates = payload.reshape(payload.shape[0], -1).view(torch.int8)
        torch_npu.npu_scatter_nd_update_(cache.view(torch.int8), indices, updates)
    for payload, cache in zip((key_scale, value_scale), kv_cache[2:]):
        torch_npu.npu_scatter_nd_update_(cache, indices, payload.reshape(-1))
    notify_kv_cache_written()


class AscendC8FIAPseudoAttentionBackendImpl(AscendC8MXFPAttentionBackendImpl):
    supports_quant_query_input = False

    def reshape_and_cache(  # type: ignore[override]
        self, quant_key, quant_value, key_scale, value_scale, kv_cache, attn_metadata
    ):
        """FIA mode-4 cache update; QFA's NZ/E8M0 scatter has a different contract."""
        scatter_fia_fp8_cache(
            quant_key,
            quant_value,
            key_scale,
            value_scale,
            kv_cache,
            attn_metadata.slot_mapping[: quant_key.shape[0]],
        )

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
        if output is None:
            raise ValueError("FIA FP8 pseudo attention requires an output buffer.")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("FIA FP8 pseudo attention does not support output quantization.")
        if attn_metadata is None:
            return output.zero_()
        if not is_950():
            raise NotImplementedError("FP8 KV pseudo quantization requires Ascend 950.")
        if self.vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("FIA FP8 pseudo cache layout does not support PD KV transfer yet.")
        if self.sliding_window is not None or getattr(self, "enable_hamming_sparse", False):
            raise NotImplementedError("FIA FP8 pseudo attention supports full attention only.")
        if query.dtype != torch.bfloat16:
            raise ValueError("FIA FP8 pseudo attention requires BF16 query.")
        if (
            len(kv_cache) != 4
            or any(cache.ndim != 3 or cache.dtype != torch.float8_e4m3fn for cache in kv_cache[:2])
            or any(cache.ndim != 2 or cache.dtype != torch.float32 for cache in kv_cache[2:])
        ):
            raise ValueError("FIA FP8 pseudo attention requires paged FP8 KV and FP32 per-token scale caches.")
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens].contiguous()
        if num_tokens == 0:
            return output
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            # Flatten only the quantizer input, so all local heads of a token
            # share one scale. Restore TND before crossing the cache API.
            quant_key, key_scale = torch_npu.npu_dynamic_quant(
                key[:num_tokens].reshape(num_tokens, -1),
                dst_type=torch.float8_e4m3fn,
                quant_mode="pertoken",
            )
            quant_value, value_scale = torch_npu.npu_dynamic_quant(
                value[:num_tokens].reshape(num_tokens, -1),
                dst_type=torch.float8_e4m3fn,
                quant_mode="pertoken",
            )
            kv_shape = (num_tokens, self.num_kv_heads, self.head_size)
            self.reshape_and_cache(
                quant_key.view(kv_shape), quant_value.view(kv_shape), key_scale, value_scale, kv_cache, attn_metadata
            )

        record_attention_compute_start()
        sink_key = getattr(layer, "sink_key", None)
        kwargs = self._fia_kwargs(query, kv_cache, attn_metadata)
        if _EXTRA_CTX.capturing:
            workspace = get_capture_resource(
                _FIA_V2_WORKSPACE_KEY,
                lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kwargs),
                self._use_max_workspace_for_fia_graph,
            )
            lse = torch.empty((num_tokens, self.num_heads, 1), dtype=torch.float32, device=query.device)
            attn_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
            register_task(
                torch_npu.npu_fused_infer_attention_score_v2.out,
                {**kwargs, "workspace": workspace, "out": [attn_output, lse]},
                FIAV2ParamProvider(self.get_graph_metadata_layer_name(layer.layer_name)),
            )
        else:
            attn_output, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        if sink_key is not None:
            # Same raw BF16 Q and learned-key sink semantics as the C16 path.
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
        if not _EXTRA_CTX.capturing:
            output[:num_tokens] = attn_output
        return output

    def _fia_kwargs(self, query, kv_cache, metadata):
        q_ends = metadata.actual_seq_lengths_q
        # FIA V5 requires defaultMask for single-query sequences. Their KV
        # lengths already exclude future tokens, so a causal mask is redundant.
        causal = metadata.causal and any(end - start > 1 for start, end in zip([0, *q_ends[:-1]], q_ends))
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
            "actual_seq_qlen": metadata.actual_seq_lengths_q,
            "actual_seq_kvlen": metadata.seq_lens_list,
            "atten_mask": metadata.attn_mask if causal else None,
            "sparse_mode": 3 if causal else 0,
            "pre_tokens": SWA_INT_MAX,
            "next_tokens": 0 if causal else SWA_INT_MAX,
            "return_softmax_lse": True,
        }
