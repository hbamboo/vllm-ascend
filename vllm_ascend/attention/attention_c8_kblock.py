# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""SWA FP8 K-block/channel cache with BF16 FIA computation."""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_c8_fia import (
    AscendC8FIAPseudoAttentionBackend,
    AscendC8FIAPseudoAttentionBackendImpl,
)
from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPMetadataBuilder
from vllm_ascend.attention.attention_v1 import _FIA_V2_WORKSPACE_KEY, AscendMetadata
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, notify_kv_cache_written
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.sink import sink_apply_triton
from vllm_ascend.utils import weak_ref_tensors


@dataclass
class AscendC8KBlockMetadata(AscendMetadata):
    kblock_num_reqs: torch.Tensor | None = None


class AscendC8KBlockMetadataBuilder(AscendC8MXFPMetadataBuilder):
    """Keep metadata addresses stable when a graph's request grouping changes."""

    metadata_cls = AscendC8KBlockMetadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kblock_capacity = vllm_config.scheduler_config.max_num_seqs + 1
        self.kblock_query_start_loc = torch.zeros(self.kblock_capacity + 1, dtype=torch.int32, device=device)
        self.kblock_seq_lens = torch.ones(self.kblock_capacity, dtype=torch.int32, device=device)
        self.kblock_block_tables = torch.zeros(
            (self.kblock_capacity, self.max_num_blocks_per_req), dtype=torch.int32, device=device
        )
        self.kblock_num_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        self.kblock_num_reqs_cpu = torch.empty(
            1, dtype=torch.int32, device="cpu", pin_memory=torch.device(device).type != "cpu"
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendC8KBlockMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_reqs = len(metadata.actual_seq_lengths_q)
        num_columns = metadata.block_tables.shape[1]
        if num_reqs > self.kblock_capacity or num_columns > self.max_num_blocks_per_req:
            raise ValueError("K-block attention metadata exceeds its configured request/block-table capacity.")

        self.kblock_num_reqs_cpu[0] = num_reqs
        self.kblock_num_reqs.copy_(self.kblock_num_reqs_cpu, non_blocking=True)
        self.kblock_query_start_loc[: num_reqs + 1].copy_(metadata.query_start_loc_gpu[: num_reqs + 1])
        num_seq_lens = min(num_reqs, metadata.seq_lens_gpu.shape[0])
        self.kblock_seq_lens[:num_seq_lens].copy_(metadata.seq_lens_gpu[:num_seq_lens])
        if num_seq_lens < num_reqs:
            # The extra padding request has a Q boundary but no runner-owned
            # sequence-length entry. Its discarded output reads a zero KV row.
            self.kblock_seq_lens[num_seq_lens:num_reqs].fill_(1)
        self.kblock_block_tables[:num_reqs, :num_columns].copy_(metadata.block_tables[:num_reqs])

        metadata.query_start_loc_gpu = self.kblock_query_start_loc
        metadata.seq_lens_gpu = self.kblock_seq_lens
        metadata.block_tables = self.kblock_block_tables
        metadata.kblock_num_reqs = self.kblock_num_reqs
        return metadata


@dataclass(frozen=True, slots=True)
class KBlockFIAParamProvider:
    layer_name: str | None
    window_size: int
    block_table: torch.Tensor = field(compare=False, hash=False, repr=False)

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]
        query_ends = metadata.actual_seq_lengths_q
        query_starts = [0, *query_ends[:-1]]
        cropped_seq_lens = [
            min(seq_len, self.window_size + end - start)
            for seq_len, start, end in zip(metadata.seq_lens_list, query_starts, query_ends)
        ]
        return {
            "actual_seq_qlen": query_ends,
            "actual_seq_kvlen": cropped_seq_lens,
            "block_table": self.block_table[: len(query_ends)],
        }


class AscendC8KBlockAttentionBackend(AscendC8FIAPseudoAttentionBackend):
    @staticmethod
    def get_impl_cls():
        AscendC8FIAPseudoAttentionBackend.get_impl_cls()
        return AscendC8KBlockAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        AscendC8FIAPseudoAttentionBackend.get_builder_cls()
        return AscendC8KBlockMetadataBuilder


class AscendC8KBlockAttentionBackendImpl(AscendC8FIAPseudoAttentionBackendImpl):
    """FP8 pages with explicit window dequantization before BF16 FIA.

    FIA's FP8 modes cannot express a distinct channel scale for each K
    token group. The FP8 cache and its scales are gathered into shared BF16
    scratch; the native attention operation still handles the SWA band.
    """

    k_block_group_size: int = 32

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
        from vllm_ascend.ops.triton.c8_kblock_cache import gather_dequant_window, update_kblock_cache

        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Block-channel SWA K does not support output quantization.")
        if attn_metadata is None:
            return output.zero_()
        ends = attn_metadata.actual_seq_lengths_q
        num_tokens = ends[-1]
        if not num_tokens:
            return output
        query = query[:num_tokens].contiguous()
        query_lens = [end - start for start, end in zip([0, *ends[:-1]], ends)]
        if _EXTRA_CTX.capturing:
            # Replaying a token bucket can change both the number of
            # requests and their Q lengths. Device active_B masks the fixed
            # capacity; no stale query-start tail can write an old page.
            capacity = min(num_tokens, self.vllm_config.scheduler_config.max_num_seqs + 1)
            query_bound = num_tokens
        else:
            capacity = len(ends)
            query_bound = max(query_lens)
        if key is not None and value is not None:
            update_kblock_cache(
                key[:num_tokens],
                value[:num_tokens],
                kv_cache,
                attn_metadata.slot_mapping,
                attn_metadata.query_start_loc_gpu,
                attn_metadata.seq_lens_gpu,
                attn_metadata.block_tables,
                num_reqs=capacity,
                active_num_reqs=attn_metadata.kblock_num_reqs,
                group_size=self.k_block_group_size,
                max_query_len=query_bound,
            )
            notify_kv_cache_written()
        record_attention_compute_start()
        block_size = kv_cache[0].shape[1]
        pages_per_request = (self.sliding_window + query_bound + block_size - 1) // block_size
        resource_key = (
            "c8_kblock_window",
            capacity,
            pages_per_request,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
        )

        def make_window():
            shape = (capacity * pages_per_request, block_size, self.num_kv_heads * self.head_size)
            return (
                torch.empty(shape, dtype=query.dtype, device=query.device),
                torch.empty(shape, dtype=query.dtype, device=query.device),
                torch.arange(capacity * pages_per_request, dtype=torch.int32, device=query.device).view(
                    capacity, pages_per_request
                ),
            )

        if _EXTRA_CTX.capturing:
            window_key, window_value, identity_blocks = get_capture_resource(resource_key, make_window)
        else:
            if resource_key not in attn_metadata.qfa_metadata_cache:
                attn_metadata.qfa_metadata_cache[resource_key] = make_window()
            window_key, window_value, identity_blocks = attn_metadata.qfa_metadata_cache[resource_key]
        gather_dequant_window(
            kv_cache,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc_gpu,
            attn_metadata.seq_lens_gpu,
            window_key,
            window_value,
            num_reqs=capacity,
            active_num_reqs=attn_metadata.kblock_num_reqs,
            window_size=self.sliding_window,
            group_size=self.k_block_group_size,
            pages_per_request=pages_per_request,
            num_tokens=num_tokens,
        )
        provider = KBlockFIAParamProvider(
            self._graph_metadata_layer_name(layer), self.sliding_window, weak_ref_tensors(identity_blocks)
        )
        kwargs = {
            "query": query,
            "key": window_key,
            "value": window_value,
            "input_layout": "TND",
            "num_query_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "softmax_scale": self.scale,
            "block_size": block_size,
            "atten_mask": attn_metadata.attn_mask,
            "sparse_mode": 4,
            "pre_tokens": self.sliding_window,
            "next_tokens": 0,
            "return_softmax_lse": True,
            **provider.resolve({provider.layer_name: attn_metadata}),
        }
        if _EXTRA_CTX.capturing:
            workspace = get_capture_resource(
                (_FIA_V2_WORKSPACE_KEY, "c8_kblock", self.sliding_window),
                lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kwargs),
                self._use_max_workspace_for_fia_graph,
            )
            lse = torch.empty((num_tokens, self.num_heads, 1), dtype=torch.float32, device=query.device)
            attn_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
            register_task(
                torch_npu.npu_fused_infer_attention_score_v2.out,
                {**kwargs, "workspace": workspace, "out": [attn_output, lse]},
                provider,
            )
        else:
            attn_output, lse = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
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
        if not _EXTRA_CTX.capturing:
            output[:num_tokens] = attn_output
        return output
