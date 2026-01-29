/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "dual_level_quant_batch_matmul_tiling_data.h"
#include "dual_level_quant_batch_matmul_tiling_key.h"

template <int SOC_VERSION_TYPE>
__global__ __aicore__ void dual_level_quant_batch_matmul(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Level0Scale, GM_ADDR x1Level1Scale, GM_ADDR x2Level0Scale, GM_ADDR x2Level1Scale,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(DualLevelQuantBatchMatmulBasicTilingData);
    if constexpr (SOC_VERSION_TYPE == DLQBMM_SOC_RESERVERD) {
    }
}