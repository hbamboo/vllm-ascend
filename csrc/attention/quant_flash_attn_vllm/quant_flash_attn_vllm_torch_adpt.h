/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QUANT_FLASH_ATTN_VLLM_TORCH_ADPT_H
#define QUANT_FLASH_ATTN_VLLM_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

constexpr int64_t QFA_DIM_THREE = 3;
constexpr int64_t QFA_MAX_DIM_SIZE = 8;
constexpr int64_t QFA_QUANT_MODE_MXFP4 = 5;
constexpr int64_t QFA_METADATA_STRIDE = 16;
constexpr int64_t QFA_METADATA_ALIGN = 4096;
constexpr int64_t QFA_DEFAULT_CUBE_CORE = 36;
constexpr int64_t QFA_DEFAULT_VECTOR_CORE = 72;

inline int64_t QfaQkvDtypeRatio(int64_t quant_mode)
{
    // MXFP4 packs two fp4 elements per byte; the logical output dim doubles.
    // The vllm QFA path only exercises MXFP8 modes, but keep the source rule.
    return quant_mode == QFA_QUANT_MODE_MXFP4 ? 2 : 1;
}

inline void QfaGetCoreNums(uint32_t &aic_core_num, uint32_t &aiv_core_num)
{
    aic_core_num = 0;
    aiv_core_num = 0;
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aic_core_num) != ACL_SUCCESS ||
        aic_core_num == 0) {
        aic_core_num = static_cast<uint32_t>(QFA_DEFAULT_CUBE_CORE);
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aiv_core_num) != ACL_SUCCESS ||
        aiv_core_num == 0) {
        aiv_core_num = static_cast<uint32_t>(QFA_DEFAULT_VECTOR_CORE);
    }
}

inline int64_t QfaCalcBatchSize(int64_t batch_size,
                                const c10::optional<at::Tensor> &seqused_q,
                                const c10::optional<at::Tensor> &cu_seqlens_q)
{
    if (seqused_q.has_value() && seqused_q->defined()) {
        return seqused_q->size(0);
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q->defined() && cu_seqlens_q->size(0) > 0) {
        return cu_seqlens_q->size(0) - 1;
    }
    return batch_size;
}

inline int64_t QfaCalcMetadataRowSize(int64_t batch_size, int64_t num_heads_q)
{
    uint32_t aic_core_num = 0;
    uint32_t aiv_core_num = 0;
    QfaGetCoreNums(aic_core_num, aiv_core_num);
    const int64_t safe_batch = batch_size > 0 ? batch_size : 1;
    const int64_t fa_size = static_cast<int64_t>(aic_core_num) * QFA_METADATA_STRIDE * safe_batch * num_heads_q;
    const int64_t fd_size = static_cast<int64_t>(aiv_core_num) * QFA_METADATA_STRIDE * safe_batch * num_heads_q;
    const int64_t schedule_size = QFA_METADATA_STRIDE + fa_size + fd_size;
    return (schedule_size + QFA_METADATA_ALIGN - 1) / QFA_METADATA_ALIGN * QFA_METADATA_ALIGN;
}

inline std::tuple<at::Tensor, at::Tensor> ConstructQfaOutputTensor(
    const at::Tensor &q,
    const at::Tensor &v,
    int64_t quant_mode,
    const std::string &layout_q,
    const std::string &layout_kv,
    const std::string &layout_out,
    bool return_softmax_lse)
{
    int64_t t_size = 0;
    int64_t n_size = 0;
    int64_t d_size = 0;
    int64_t s_size = 0;
    int64_t b_size = 0;
    at::SmallVector<int64_t, QFA_MAX_DIM_SIZE> attention_out_size;
    at::SmallVector<int64_t, QFA_MAX_DIM_SIZE> softmax_out_size;

    if (layout_q == "TND") {
        t_size = q.size(0);
        n_size = q.size(1);
    } else if (layout_q == "NTD") {
        n_size = q.size(0);
        t_size = q.size(1);
    } else if (layout_q == "BSND") {
        b_size = q.size(0);
        s_size = q.size(1);
        n_size = q.size(2);
    } else {
        b_size = q.size(0);
        n_size = q.size(1);
        s_size = q.size(2);
    }
    if (layout_kv == "TND") {
        d_size = v.size(2);
    } else if (layout_kv == "PA_NZ") {
        d_size = v.size(2) * v.size(4);
    } else {
        d_size = v.size(v.dim() - 1);
    }

    if (return_softmax_lse) {
        if (q.dim() == QFA_DIM_THREE) {
            softmax_out_size = {n_size, t_size};
        } else {
            softmax_out_size = {b_size, n_size, s_size};
        }
    } else {
        softmax_out_size = {0};
    }
    at::Tensor softmax_lse = at::empty(softmax_out_size, q.options().dtype(at::kFloat));

    const int64_t q_dtype_ratio = QfaQkvDtypeRatio(quant_mode);
    if (layout_out == "TND") {
        attention_out_size = {t_size, n_size, q_dtype_ratio * d_size};
    } else if (layout_out == "BNSD") {
        attention_out_size = {b_size, n_size, s_size, q_dtype_ratio * d_size};
    } else {
        attention_out_size = {b_size, s_size, n_size, q_dtype_ratio * d_size};
    }
    at::Tensor attention_out = at::empty(attention_out_size, q.options().dtype(at::kBFloat16));
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

inline at::Device ResolveQfaMetadataDevice(
    c10::string_view device,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv)
{
    if (seqused_kv.has_value() && seqused_kv->defined()) {
        return seqused_kv->device();
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q->defined()) {
        return cu_seqlens_q->device();
    }
    if (seqused_q.has_value() && seqused_q->defined()) {
        return seqused_q->device();
    }
    return at::Device(std::string(device));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_quant_flash_attn_vllm(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &q_descale,
    const at::Tensor &k_descale,
    const at::Tensor &v_descale,
    int64_t quant_mode,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata,
    double softmax_scale,
    int64_t mask_mode,
    int64_t win_left,
    int64_t win_right,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    c10::string_view layout_q,
    c10::string_view layout_q_descale,
    c10::string_view layout_kv,
    c10::string_view layout_out,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.");
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.");
    TORCH_CHECK(v.numel() > 0, "Tensor v is empty.");

    std::string layout_q_str(layout_q);
    std::string layout_q_descale_str(layout_q_descale);
    std::string layout_kv_str(layout_kv);
    std::string layout_out_str(layout_out);

    auto outputs = ConstructQfaOutputTensor(q, v, quant_mode, layout_q_str, layout_kv_str, layout_out_str,
                                            return_softmax_lse);
    at::Tensor attention_out = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_q_descale_ptr = const_cast<char *>(layout_q_descale_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    EXEC_NPU_CMD(
        aclnnQuantFlashAttnVllm,
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        quant_mode,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q_ptr,
        layout_q_descale_ptr,
        layout_kv_ptr,
        layout_out_ptr,
        return_softmax_lse,
        attention_out,
        softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

at::Tensor npu_quant_flash_attn_vllm_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t head_dim_v,
    int64_t mask_mode,
    int64_t win_left,
    int64_t win_right,
    c10::string_view layout_q,
    c10::string_view layout_q_descale,
    c10::string_view layout_kv,
    c10::string_view layout_out,
    bool is_grad_enabled,
    c10::string_view device)
{
    const int64_t real_batch = QfaCalcBatchSize(batch_size, seqused_q, cu_seqlens_q);
    const int64_t real_head_dim_v = head_dim_v < 0 ? head_dim : head_dim_v;

    at::Device output_device = ResolveQfaMetadataDevice(device, cu_seqlens_q, seqused_q, seqused_kv);
    const int64_t row_size = QfaCalcMetadataRowSize(real_batch, num_heads_q);
    at::Tensor metadata = at::empty({2, row_size}, at::TensorOptions().dtype(at::kInt).device(output_device));

    std::string layout_q_str(layout_q);
    std::string layout_q_descale_str(layout_q_descale);
    std::string layout_kv_str(layout_kv);
    std::string layout_out_str(layout_out);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_q_descale_ptr = const_cast<char *>(layout_q_descale_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    EXEC_NPU_CMD(
        aclnnQuantFlashAttnVllmMetadata,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        real_batch,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        real_head_dim_v,
        quant_mode,
        mask_mode,
        win_left,
        win_right,
        layout_q_ptr,
        layout_q_descale_ptr,
        layout_kv_ptr,
        layout_out_ptr,
        is_grad_enabled,
        metadata);
    return metadata;
}

}  // namespace vllm_ascend

#endif  // QUANT_FLASH_ATTN_VLLM_TORCH_ADPT_H
