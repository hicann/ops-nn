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
 * \file gelu_grad.cpp
 * \brief
 */

#include "gelu_grad.h"

using namespace NsGeluGrad;

enum class GeluGradTilingKey : uint32_t
{
    TILING_KEY_910B = 0,
    TILING_KEY_310B = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluGradTilingData, tiling_data, tiling);

    if(TILING_KEY_IS(0))
    {
        KernelGeluGrad<DTYPE_DY, DTYPE_X, DTYPE_Z, true> op;
        op.Init(dy, x, y, z, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, tiling_data.versionNum);
        op.Process();
    }
    else if(TILING_KEY_IS(1))
    {
        KernelGeluGrad<DTYPE_DY, DTYPE_X, DTYPE_Z, false> op;
        op.Init(dy, x, y, z, tiling_data.smallCoreDataNum,
                tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
                tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
                tiling_data.tailBlockNum, tiling_data.versionNum);
        op.Process();
    }
}

