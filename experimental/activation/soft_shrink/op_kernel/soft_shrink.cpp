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
 * \file soft_shrink.cpp
 * \brief
 */

#include "soft_shrink.h"


template <uint32_t schMode>
__global__ __aicore__ void soft_shrink(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftShrinkTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftShrinkTilingData, tilingData, tiling);
    NsSoftShrink::KernelSoftShrink<DTYPE_X> op; // 算子kernel实例获取
    op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
       tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.lambd);
    op.Process();
}
