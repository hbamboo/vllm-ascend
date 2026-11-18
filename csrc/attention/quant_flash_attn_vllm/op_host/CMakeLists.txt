# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

set(QUANT_FLASH_ATTN_VLLM_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for quant_flash_attn_vllm")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        quant_flash_attn_vllm_def.cpp
    )
endif()

# 依赖 attention/common 的头文件已通过算子目录内的 common/ 私有快照与
# qfa_kernel_deps.h / qfa_host_deps.h 两个聚合头收敛，不再声明 attention/common 依赖。

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME QuantFlashAttnVllm
        COMPUTE_UNIT ascend950
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

file(GLOB_RECURSE ARCH35_CHECKER_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/checkers/*checker*.cpp
)

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME QuantFlashAttnVllm
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_ops_compile_options(
            OP_NAME QuantFlashAttnVllm
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ${ARCH35_CHECKER_SRC_FILES}
    qfa_tiling_info_parser.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE quant_flash_attn_vllm
        ACLNNTYPE aclnn_inner
    )
endif()
