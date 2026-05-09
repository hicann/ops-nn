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
 * \file gelu_grad_v2.cpp
 * \brief
 */

#include "gelu_grad_v2.h"


template <uint32_t schMode>
__global__ __aicore__ void gelu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluGradV2TilingData, tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        NsGeluGradV2::KernelGeluGradV2<DTYPE_DY> op; // 算子kernel实例获取
        op.Init(dy, x, z, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
        tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.bufferOpen);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NsGeluGradV2Tanh::KernelGeluGradV2Tanh<DTYPE_DY> op; // 算子kernel实例获取
        op.Init(dy, x, z, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
        tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.bufferOpen);
        op.Process();
    }
}
