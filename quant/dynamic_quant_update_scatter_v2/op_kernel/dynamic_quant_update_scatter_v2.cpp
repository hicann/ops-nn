/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_update_scatter_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_v2_db.h"
#include "dynamic_quant_update_scatter_v2.h"

using namespace AscendC;
using namespace DynamicQuantUpdateScatterV2NDOpt;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter_v2(
    GM_ADDR x, GM_ADDR indices, GM_ADDR var, GM_ADDR varScale, GM_ADDR varOffset, GM_ADDR varOut, GM_ADDR varScaleOut,
    GM_ADDR varOffsetOut, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (x == nullptr || indices == nullptr || var == nullptr || varScale == nullptr || varOffset == nullptr) {
        return;
    }

    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.coreNum) {
        return;
    }
    TPipe pipe;
    if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        DynamicQuantUpdateScatterV2<DTYPE_X, DTYPE_VAR> op(&pipe);
        op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
        DynamicQuantUpdateScatterV2DbOpt<DTYPE_X, DTYPE_VAR> op(&pipe);
        op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
        op.Process();
    } else {
    }
}