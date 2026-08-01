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
 * \file dynamic_quant_update_scatter.cpp
 * \brief DynamicQuantUpdateScatter RegBase kernel for Ascend 950.
 */

#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_regbase.h"
#include "dynamic_quant_update_scatter_tiling_data.h"

using namespace AscendC;
using namespace DynamicQuantUpdateScatterND;

#define TILING_KEY_REGBASE_NO_SMOOTH 0
#define TILING_KEY_REGBASE_WITH_SMOOTH 1

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices,
                                                                   GM_ADDR updates, GM_ADDR smoothScales,
                                                                   GM_ADDR varOut, GM_ADDR varScaleOut,
                                                                   GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DynamicQuantUpdateScatterRegbaseTilingData);
    if (var == nullptr || varScale == nullptr || indices == nullptr || updates == nullptr || tiling == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(DynamicQuantUpdateScatterRegbaseTilingData, tilingData, tiling);
    if (TILING_KEY_IS(TILING_KEY_REGBASE_NO_SMOOTH)) {
        DynamicQuantUpdateScatterRegbase<DTYPE_INDICES, DTYPE_UPDATES, false> op;
        op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_REGBASE_WITH_SMOOTH)) {
        DynamicQuantUpdateScatterRegbase<DTYPE_INDICES, DTYPE_UPDATES, true> op;
        op.Init(var, varScale, indices, updates, smoothScales, &tilingData);
        op.Process();
    }
}
