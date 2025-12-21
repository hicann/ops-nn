/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_linear_cross_entropy_loss_grad_config.h
 * \brief
 */
#ifndef FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_CONFIG_H
#define FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_CONFIG_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FusedLinearCrossEntropyLossGradNS {

constexpr MatmulConfig MM_MDL_CFG = GetMDLConfig(
    false,  // intrinsicsLimit
    false, 0,
    false,  // isVecND2NZ，false
    false, false,
    false,  // enUnitFlag，false
    true, true, false, false,
    false  // enableKdimReorderLoad，false
);
constexpr MatmulConfig MM_MDL_HUGE_CFG = GetMDLConfig(
    true,  // intrinsicsLimit
    false, 0,
    false,  // isVecND2NZ
    false, false,
    false,  // enUnitFlag
    true, true, false, false,
    false  // enableKdimReorderLoad
);

#define HIGH_PERF_KEY 100UL  // 高性能分支tiling key
#define MEM_FRIENDLY_KEY 101UL  // 省显存分支tiling key
constexpr uint32_t IN_BYTE_SIZE = 4;
constexpr uint32_t OUT_BYTE_SIZE = 2;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SIZE_2 = 2;
constexpr uint32_t SIZE_3 = 3;
constexpr uint32_t SIZE_8 = 8;
constexpr uint32_t SIZE_MAX_AIC_NUM = 30;
constexpr uint32_t SIZE_32 = 32;
constexpr uint32_t SIZE_64 = 64;
constexpr uint32_t SIZE_BASE_TASK_MAX = 65536 * 3;
constexpr uint32_t OUT_SIZE_PER_32B = SIZE_32 / OUT_BYTE_SIZE;
constexpr uint32_t MAX_WS_NUM = 100;
constexpr uint8_t CVModeId = 2;
constexpr float NEG = -1.0f;

#ifdef __CCE_KT_TEST__
void EmptyTestFunc() {};
#endif  // __CCE_KT_TEST__

}  // namespace FusedLinearCrossEntropyLossGradNS

#endif  // FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_CONFIG_H