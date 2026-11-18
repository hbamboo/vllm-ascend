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
 * \file qfa_host_deps.h
 * \brief QuantFlashAttnVllm host tiling 侧唯一依赖聚合头。
 *
 * 收敛上游 ops-transformer attention/common 的 op_host 依赖：
 *   - 通用 tiling 框架命名空间（复用 vllm 仓已有的 tiling_base/tiling_templates_registry.h）
 *   - FIA tiling 基类与模板注册（来自同目录 common/ 私有快照）
 */

#ifndef QFA_HOST_DEPS_H
#define QFA_HOST_DEPS_H

#include "tiling_base/tiling_templates_registry.h"
#include "common/op_host/fia_tiling_base.h"
#include "common/op_host/fia_tiling_templates_registry.h"

#endif // QFA_HOST_DEPS_H
