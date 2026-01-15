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
 * \file mish_grad.cpp
 * \brief
 */

#include "mish_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void mish_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MishGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(MishGradTilingData, tilingData, tiling);
    NsMishGrad::KernelMishGrad<DTYPE_X> op;
     op.Init(grad, x, tanhx, x_grad, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
        tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.haveTanhx);
    op.Process();
}
