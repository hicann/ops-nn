/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh_grad.cpp
 * \brief
 */

#include "tanh_grad_tiling_key.h"
#include "tanh_grad.h"
using namespace AscendC;

template <uint32_t schMode>
__global__ __aicore__ void tanh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhGradTilingData, tilingData, tiling);
    NsTanhGrad::KernelTanhGrad<DTYPE_Y> op;
    op.Init(y, dy, dx, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum,
            tilingData.finalSmallTileNum, tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum);
    op.Process();
}
