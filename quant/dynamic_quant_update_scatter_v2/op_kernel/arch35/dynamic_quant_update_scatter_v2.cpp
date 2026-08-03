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
 * \file dynamic_quant_update_scatter_v2.cpp
 * \brief DynamicQuantUpdateScatterV2 RegBase kernel entry for Ascend 950.
 */

#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_v2_regbase.h"
#include "dynamic_quant_update_scatter_v2_tiling_data.h"

using namespace AscendC;
using namespace DynamicQuantUpdateScatterV2ND;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter_v2(GM_ADDR x, GM_ADDR indices, GM_ADDR var,
                                                                      GM_ADDR varScale, GM_ADDR varOffset,
                                                                      GM_ADDR varOut, GM_ADDR varScaleOut,
                                                                      GM_ADDR varOffsetOut, GM_ADDR workSpace,
                                                                      GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DynamicQuantUpdateScatterV2RegbaseTilingData);
    if (x == nullptr || indices == nullptr || var == nullptr || varScale == nullptr || varOffset == nullptr ||
        tiling == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(DynamicQuantUpdateScatterV2RegbaseTilingData, tilingData, tiling);
    DynamicQuantUpdateScatterV2Regbase<DTYPE_X, DTYPE_VAR> op;
    op.Init(x, indices, var, varScale, varOffset, &tilingData);
    op.Process();
}
