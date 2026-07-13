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
 * \file max_pool_v3.cpp
 * \brief max_pool_v3 kernel dispatch
 */

#include "max_pool_v3.h"

using namespace NsMaxPoolV3;

template <uint64_t schMode>
__global__ __aicore__ void max_pool_v3(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(MaxPoolV3TilingData, tilingData, tiling);

    // Compute window size from tiling data
    uint32_t windowSize = tilingData.kH * tilingData.kW;

    KernelMaxPoolV3<DTYPE_X> op;
    op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum,
            tilingData.finalSmallTileNum, tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.n, tilingData.c, tilingData.hIn,
            tilingData.wIn, tilingData.hOut, tilingData.wOut, tilingData.kH, tilingData.kW, tilingData.sH,
            tilingData.sW, tilingData.padT, tilingData.padL, windowSize);
    op.Process();
}
