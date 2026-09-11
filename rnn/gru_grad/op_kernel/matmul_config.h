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
 * \file matmul_config.h
 * \brief gru_grad matmul 配置: 默认配置与超大内轴配置
 */

#ifndef GRU_GRAD_MATMUL_CONFIG_H
#define GRU_GRAD_MATMUL_CONFIG_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// 默认配置: 与原始 gru_grad (默认 CFG_NORM) 行为一致
constexpr MatmulConfig MM_CFG = GetNormalConfig(false // intrinsicsLimit
);
// 大内轴配置: intrinsicsLimit=true 使能循环执行 GM→L1 数据搬入,
// 当单核内轴 >= 65535 时启用,小 shape 无副作用
constexpr MatmulConfig MM_HUGE_CFG = GetNormalConfig(true // intrinsicsLimit
);

#endif // GRU_GRAD_MATMUL_CONFIG_H
