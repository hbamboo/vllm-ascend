import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm_ascend.ascend_config import get_ascend_config

from ..base import AscendAttentionScheme


def _quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        # ModelSlim exports the per-channel V cache scale as a column vector
        # ([hidden, 1]); flatten it first. The loader delivers the FULL-width
        # scale on every rank (attention-module parameters bypass the
        # ColumnParallelLinear sharding), so slice it here following the same
        # rule vLLM uses to replicate KV heads under GQA TP: with
        # num_kv_heads < tp_size each head is replicated across
        # tp_size // num_kv_heads ranks and rank r owns the slice of
        # head r // (tp_size // num_kv_heads); otherwise a plain narrow.
        # Some recipes store one set of per-channel scales shared by every KV
        # head; tile it out to this rank's head count first.
        if loaded_weight.dim() != 1:
            loaded_weight = loaded_weight.flatten()
        if loaded_weight.numel() < param.numel() and param.numel() % loaded_weight.numel() == 0:
            loaded_weight = loaded_weight.repeat(param.numel() // loaded_weight.numel())
        if loaded_weight.numel() != param.numel():
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            head_dim = param.numel()  # this rank's full (possibly replicated) head width
            total = loaded_weight.numel()
            num_heads_total = total // head_dim if total % head_dim == 0 else None
            if num_heads_total is None or num_heads_total < 1:
                raise AssertionError(
                    "[vllm-ascend/MXFP8_PER_CHANNEL] Cannot map V cache scale of "
                    f"{total} elements onto a per-rank head width of {head_dim} "
                    f"(TP size {tp_size}, TP rank {tp_rank})."
                )
            heads_per_rank_group = max(1, tp_size // num_heads_total)
            src_head = tp_rank // heads_per_rank_group
            loaded_weight = loaded_weight.narrow(0, src_head * head_dim, head_dim)
        assert param.numel() == loaded_weight.numel(), (
            "[vllm-ascend/MXFP8_PER_CHANNEL] V cache scale size mismatch: parameter "
            f"has {param.numel()} elements but the sliced weight has "
            f"{loaded_weight.numel()} (TP size "
            f"{get_tensor_model_parallel_world_size()}, TP rank "
            f"{get_tensor_model_parallel_rank()})."
        )
        param.data.copy_(loaded_weight.view_as(param))


class AscendC8MXFPKVCacheAttentionMethod(AscendAttentionScheme):
    """MXFP8 KV cache storage for dense-attention models.

    K/V are cached as FP8 E4M3 and their E8M0 scales are stored in extra
    cache tensors: K uses dynamic per-token-group scales written at scatter
    time, V uses the static per-channel E8M0 scale stored in the ModelSlim
    checkpoint. QFA FP8 uses FP32 Q/K scales per token with dynamic per-head
    V, or per token/head with checkpoint static per-channel V.
    Enabled with ``--kv-cache-dtype mxfp8``.
    """

    v_quant_mode = "per_channel"
    q_quant_mode = "per_token_group"

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.float8_e4m3fn
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_c8_mxfp import (
                MXFP_MAX_QUANT_BLOCK_SIZE,
                AscendC8MXFPAttentionBackend,
                AscendC8MXFPAttentionBackendImpl,
            )

            config = get_ascend_config()
            self.q_quant_mode = config.c8_q_quant_mode
            self.v_quant_mode = config.c8_v_quant_mode
            # The delivered QFA supports SWA in MXFP8 mode 1 only. Keep the
            # configured full-attention scheme; use PR #13's scheme for SWA.
            sliding_mxfp = getattr(layer, "sliding_window", None) is not None and self.q_quant_mode in (
                "per_token",
                "per_token_head",
            )
            if sliding_mxfp:
                self.q_quant_mode = "per_token_group"
                self.v_quant_mode = "per_channel"
            if self.q_quant_mode in ("per_token", "per_token_head"):
                from vllm_ascend.attention.attention_c8_mxfp import AscendC8QFAFP8AttentionBackend

                if (layer.head_size, layer.head_size_v) != (128, 128):
                    raise ValueError("QFA FP8 per-token Q/K requires K/V head dimensions of 128.")
                layer.attn_backend = AscendC8QFAFP8AttentionBackend
                layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
                layer.impl.supports_quant_query_input = False
            elif self.v_quant_mode == "per_token":
                from vllm_ascend.attention.attention_c8_fia import (
                    AscendC8FIAPseudoAttentionBackend,
                    AscendC8FIAPseudoAttentionBackendImpl,
                )

                if layer.head_size != layer.head_size_v:
                    raise ValueError("Dynamic per-token KV requires equal K/V head dimensions.")
                layer.attn_backend = AscendC8FIAPseudoAttentionBackend
                layer.impl.__class__ = AscendC8FIAPseudoAttentionBackendImpl
            else:
                layer.attn_backend = AscendC8MXFPAttentionBackend
                layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
            # Changing __class__ does not invoke the new class's __init__, so
            # initialize the state the impl relies on here.
            layer.impl.enable_hamming_sparse = False
            layer.impl.q_quant_mode = self.q_quant_mode
            layer.impl.k_quant_mode = "per_token_group" if sliding_mxfp else config.c8_k_quant_mode
            layer.impl.v_quant_mode = self.v_quant_mode
            # Reserve equal full-layer pages even when tracing selects a subset.
            offset_mode = getattr(config, "c8_k_offset_mode", "none")
            layer.impl.k_offset_reserve = offset_mode != "none" and not sliding_mxfp
            layer.impl.k_offset_mode = offset_mode if layer.impl.k_offset_reserve else "none"
            offset_layers = getattr(config, "c8_k_offset_layers", None)
            if layer.impl.k_offset_reserve and offset_layers is not None:
                from vllm.model_executor.models.utils import extract_layer_index

                if extract_layer_index(self.prefix) not in offset_layers:
                    layer.impl.k_offset_mode = "none"
            if (
                self.q_quant_mode not in ("per_token", "per_token_head")
                and layer.impl.k_quant_mode == "per_token"
                and layer.num_kv_heads * layer.head_size > MXFP_MAX_QUANT_BLOCK_SIZE
            ):
                raise ValueError("Per-token C8 K requires at most 1024 local KV channels; increase tensor parallelism.")

        # Keep the checkpoint parameter even when runtime V scales are dynamic.
        hidden_size = layer.num_kv_heads * layer.head_size_v
        weight_param = torch.nn.Parameter(
            torch.full((hidden_size,), 127, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("v_cache_scale", weight_param)
        # When loading weights, segment them according to TP
        weight_param.weight_loader = _quant_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.v_quant_mode == "per_head":
            initial_scale = torch.finfo(torch.float32).tiny
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            device = layer.v_cache_scale.device
            layer.impl.qfa_v_head_bound = torch.full(
                (layer.num_kv_heads,), initial_scale * fp8_max, dtype=torch.bfloat16, device=device
            )
            layer.impl.qfa_v_descale = torch.full(
                (layer.num_kv_heads,), initial_scale, dtype=torch.float32, device=device
            )
            layer.impl.qfa_p_scale = torch.tensor([fp8_max], dtype=torch.float32, device=device)
            return
        if self.v_quant_mode == "per_token":
            # Load fa_v.scale for checkpoint compatibility; runtime generates
            # and caches a fresh scale for every V token instead.
            return
        vllm_config = get_current_vllm_config()
        target_dtype = vllm_config.model_config.dtype
        raw = layer.v_cache_scale.data
        # A minmax calibrator emits 0 for a channel whose absmax was 0;
        # sanitize to the neutral 127 so both consumers (the reciprocal below
        # and the raw bytes broadcast into the V-scale cache) stay neutral.
        if bool((raw == 0).any()):
            raw[raw == 0] = 127
        exponent = raw.to(torch.float32) - 127
        descale = torch.exp2(exponent)
        # Only the reciprocal is consumed (npu_quantize needs 1/scale); the
        # raw E8M0 bytes are broadcast into the V-scale cache as-is.
        layer.v_cache_scale_float_reciprocal = torch.nn.Parameter(
            (1 / descale).to(target_dtype),
            requires_grad=False,
        )
        if self.q_quant_mode == "per_token_head":
            layer.impl.qfa_v_descale = torch.ones(layer.num_kv_heads, dtype=torch.float32, device=raw.device)
            layer.impl.qfa_v_output_scale = descale.reshape(layer.num_kv_heads, layer.head_size_v)
            layer.impl.qfa_p_scale = torch.tensor(
                [torch.finfo(torch.float8_e4m3fn).max], dtype=torch.float32, device=raw.device
            )

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendC8MXFPKVCacheAttentionMethod.apply should not be called. "
            "C8_MXFP KV cache quantization is handled by the attention backend."
        )
