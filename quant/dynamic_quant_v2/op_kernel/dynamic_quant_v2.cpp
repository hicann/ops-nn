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
 * \file dynamic_quant_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "../dynamic_quant/dynamic_quant.h"
#if __CCE_AICORE__ < 220
#include "../dynamic_quant/dynamic_quant_align_310p.h"
#include "../dynamic_quant/dynamic_quant_unalign_310p.h"
#include "../dynamic_quant/dynamic_quant_unalign_large_310p.h"
#endif
#include "../dynamic_quant/dynamic_quant_db.h"
#include "../dynamic_quant/dynamic_quant_large_shape_opt.h"
#include "../dynamic_quant/dynamic_quant_moe.h"
#include "../dynamic_quant/dynamic_quant_moe_large_shape.h"

using namespace AscendC;
using namespace DynamicQuantNDOpt;

extern "C" __global__ __aicore__ void dynamic_quant_v2(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
    GM_ADDR tiling)
{
    if (x == nullptr || y == nullptr || scale == nullptr) {
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
#if __CCE_AICORE__ < 220
    if (TILING_KEY_IS(10100)) {
        DynamicQuantAlign310p op;
        op.Init(x, y, scale, offset, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11000)) {
        DynamicQuantUnalign310P op;
        op.Init(x, y, scale, offset, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11001)) {
        DynamicQuantUnalignLarge310P op;
        op.Init(x, y, scale, offset, &tilingData);
        op.Process();
    }
#else
    if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        DynamicQuant<DTYPE_X, DTYPE_Y> op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
        DynamicQuantDbOpt<DTYPE_X, DTYPE_Y> op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        DynamicQuantLargeShapeOpt<DTYPE_X, DTYPE_Y> op(&pipe);
        op.Init(x, smooth_scales, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        DynamicQuantMoe<DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        DynamicQuantMoeLargeShape<DTYPE_X, DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, offset, workSpace, &tilingData);
        op.Process();
    }
#endif
    else {
    }
}