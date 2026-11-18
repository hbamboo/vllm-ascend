/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qfa_kernel_deps.h
 * \brief QuantFlashAttnVllm kernel 侧唯一依赖聚合头。
 *
 * 该文件把上游 ops-transformer attention/common 的全部 kernel 依赖收敛到一处，
 * 只允许 op_kernel/quant_flash_attn_vllm.cpp 在**所有 QFA 头文件之前** include 一次。
 * 其余 op_kernel 子头文件不再各自 include common 头（依赖本聚合头先展开）。
 *
 * 依赖来源为 ops-transformer feat/qfa-sliding-window-mxfp8 分支
 * （commit 3ff306012），快照位于同目录 common/ 下，内容与上游逐字一致。
 *
 * 注意：上游多个头文件依赖“同 TU 中更早 include 的头”引入的命名空间
 * （regbaseutil / FaVectorApi::castTrait* / AscendC::Reg 等），因此本文件的
 * include 顺序按上游 kernel 入口真实包含顺序排列，不可随意调整。
 */

#ifndef QFA_KERNEL_DEPS_H
#define QFA_KERNEL_DEPS_H

// 1) CANN 基础能力（原 kernel 入口开头的几个头文件）
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "adv_api/activation/softmax.h"

// 2) 先占住 FLASH_ATTENTION_UTIL_H guard，确保拿到上游完整版 util.h
//    （vllm 仓 csrc/common/include/kernel/util.h 为裁剪版，缺少 LAYOUT_NTD 等枚举）
#include "common/include/util.h"

// 3) attention/common kernel 依赖（快照）。顺序与上游 block_cube/block_vec 一致：
//    matmul.h 引入 using namespace AscendC；
//    infer_flash_attention_comm_arch35.h -> attenmask_arch35.h 引入 AscendC::Reg；
//    flash_attention_score_common_regbase_arch35.h 引入 regbaseutil；
//    vf_mul_sel_* 引入 vf_basic_block_utils.h（FaVectorApi::castTrait*），
//    因此 vf_div_cast_arch35.h 必须排在其后。
#include "common/op_kernel/offset_calculator.h"
#include "common/op_kernel/matmul.h"
#include "common/op_kernel/FixpipeOut.h"
#include "common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "common/op_kernel/arch35/util_regbase.h"
#include "common/op_kernel/vector_common.h"

// memory_copy 依赖链（原 memory_copy_arch35_quant_flash_attn.h 的顺序）
#include "common/op_kernel/memcopy/offset_calculator_v2.h"
#include "common/op_kernel/memcopy/parser.h"
#include "common/op_kernel/memcopy/gm_coord.h"
#include "common/op_kernel/memcopy/gm_layout.h"
#include "common/op_kernel/memcopy/fa_gm_tensor.h"
#include "common/op_kernel/memcopy/fa_l1_tensor.h"
#include "common/op_kernel/memcopy/fa_ub_tensor.h"
#include "common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "common/op_kernel/memcopy/attn_copy_gm_to_ub.h"
#include "common/op_kernel/memcopy/copy_ub_to_gm.h"

// VF 依赖链（原 block_vec 的顺序：mul_sel -> flashupdate -> div_cast -> flash_decode）
#include "common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"

// attenmask_gs1 依赖 regbaseutil 命名空间（由 flash_attention_score_common 引入），
// 必须放在其后。
#include "common/op_kernel/arch35/attenmask_gs1_arch35.h"

#endif // QFA_KERNEL_DEPS_H
