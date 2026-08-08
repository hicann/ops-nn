/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool3_d.h"
#include "max_pool3_d_tiling_key.h"

template <uint32_t kernelMode>
__aicore__ inline void RunWithTiling(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MaxPool3DTilingData, tilingData, tiling);
    MaxPool3DExp::MaxPool3DKernel<DTYPE_X, kernelMode> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

template <uint32_t schMode>
__aicore__ inline void RunFeatureSchedule(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    if constexpr (schMode == MAX_POOL3_D_TPL_SCH_MODE_NCDHW_SMALL_DEPTH_STRIDE2 ||
                  schMode == MAX_POOL3_D_TPL_SCH_MODE_NCDHW_POOL2_FEATURE) {
        RunWithTiling<MAX_POOL3_D_TPL_SCH_MODE_NCDHW_STRIDE2>(x, y, tiling);
    } else {
        RunWithTiling<schMode>(x, y, tiling);
    }
}

template <uint32_t schMode>
__global__ __aicore__ void max_pool3_d(GM_ADDR x, GM_ADDR y, [[maybe_unused]] GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPool3DTilingData);
    if constexpr (schMode <= MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0) {
        RunWithTiling<schMode>(x, y, tiling);
    } else {
        RunFeatureSchedule<schMode>(x, y, tiling);
    }
}
