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
 * \file swish.cpp
 * \brief
 */

#include "swish.h"

using namespace NsSwish;

template <uint64_t schMode, uint64_t attrWork>
__global__ __aicore__ void swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwishTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwishTilingData, tiling_data, tiling);

    KernelSwish<DTYPE_X, DTYPE_X> op;
    op.Init(x, tiling_data.scale, y, tiling_data.smallCoreDataNum,
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
        tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
        tiling_data.tailBlockNum);
    op.Process();
}
