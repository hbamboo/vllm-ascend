# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""BF16 Q and dynamic per-token FP8 K/V, dequantized inside FIA v2.

Also hosts the sliding-window variant of the same cache format. Sliding
layers keep the C8 dense FP8 K/V page and drive FIA v2's BAND mask
(``sparse_mode=4``) with ``pre_tokens == sliding_window``, which is the
same control the BF16 IQuest-Q1 attention uses. FIA's antiquant path
rejects BAND when every query is one row long, so pure-decode steps feed
two query rows per request (the extra row sits one position earlier and is
discarded); verify/prefill steps use BAND directly.
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

    def _duplicate_decode_rows(self, attn_metadata) -> bool:
        """Whether this step needs the two-row decode workaround.

        FIA's antiquant path rejects BAND at ``S == 1``. Duplicating every
        single-row query keeps ``sparse_mode=4`` legal and exact: with TND
        the rows are right-aligned to the KV end, so the second row is the
        real token and the first one is one position earlier (discarded).
        """
        ends = attn_metadata.actual_seq_lengths_q
        return (
            getattr(self, "sliding_window", None) is not None
            and bool(attn_metadata.causal)
            and bool(ends)
            and ends[-1] == len(ends)
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
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("FIA per-token KV does not support output quantization.")
        if attn_metadata is None:
            return output.zero_()
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens].contiguous()
        if num_tokens == 0:
            return output
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key8, key_scale = torch_npu.npu_dynamic_quant(
                key[:num_tokens].reshape(num_tokens, -1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken"
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

        duplicate = self._duplicate_decode_rows(attn_metadata)
        if duplicate:
            fia_query = query.repeat_interleave(2, dim=0)
            fia_qlen = [2 * (i + 1) for i in range(len(attn_metadata.actual_seq_lengths_q))]
        else:
            fia_query = query
            fia_qlen = attn_metadata.actual_seq_lengths_q
        num_fia_tokens = fia_query.shape[0]
        kwargs = self._fia_kwargs(
            fia_query,
            kv_cache,
            attn_metadata,
            fia_qlen,
            sliding_band=getattr(self, "sliding_window", None) is not None and bool(attn_metadata.causal),
        )
        if _EXTRA_CTX.capturing:
            workspace = get_capture_resource(
                _FIA_V2_WORKSPACE_KEY,
                lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kwargs),
                self._use_max_workspace_for_fia_graph,
            )
            lse = torch.empty(
                (num_fia_tokens, self.num_heads, 1), dtype=torch.float32, device=query.device
            )
            if duplicate:
                # BAND decode rows are gathered back to one row per request
                # after the task; the extra row needs its own graph buffer.
                attn_output = torch.empty(
                    (num_fia_tokens, self.num_heads, self.head_size),
                    dtype=query.dtype,
                    device=query.device,
                )
            else:
                attn_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
            register_task(
                torch_npu.npu_fused_infer_attention_score_v2.out,
                {**kwargs, "workspace": workspace, "out": [attn_output, lse]},
                FIAV2ParamProvider(
                    self._graph_metadata_layer_name(layer),
                    duplicate_single_query=duplicate,
                ),
            )
        else:
            attn_output, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
            attn_output = attn_output.view(num_fia_tokens, self.num_heads, self.head_size)
            if lse is not None:
                lse = lse.view(num_fia_tokens, self.num_heads, 1)
        if duplicate:
            attn_output = attn_output[1::2].contiguous()
            if lse is not None:
                lse = lse[1::2].contiguous()
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
        if _EXTRA_CTX.capturing:
            if duplicate:
                output[:num_tokens].copy_(attn_output)
        else:
            output[:num_tokens] = attn_output
        return output

    def _fia_kwargs(self, query, kv_cache, metadata, seq_qlen, sliding_band):
        ends = metadata.actual_seq_lengths_q
        causal = metadata.causal and any(end - start > 1 for start, end in zip([0, *ends[:-1]], ends))
        kwargs = {
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
            "actual_seq_qlen": seq_qlen,
            "actual_seq_kvlen": metadata.seq_lens_list,
            "return_softmax_lse": True,
        }
        if sliding_band:
            # Same band as the BF16 sliding layers: pre_tokens counts the
            # token itself, next_tokens=0 makes the band causal.
            kwargs.update(
                atten_mask=metadata.attn_mask,
                sparse_mode=4,
                pre_tokens=self.sliding_window,
                next_tokens=0,
            )
        else:
            kwargs.update(
                atten_mask=metadata.attn_mask if causal else None,
                sparse_mode=3 if causal else 0,
                pre_tokens=SWA_INT_MAX,
                next_tokens=0 if causal else SWA_INT_MAX,
            )
        return kwargs
