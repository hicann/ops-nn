/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file silu_grad.cpp
 */
#include "silu_grad.h"

using NsSiluGrad::KernelSiluGrad;

template <uint32_t schMode>
__global__ __aicore__ void silu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)schMode;
    (void)workspace;
    REGISTER_TILING_DEFAULT(SiluGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SiluGradTilingData, tilingData, tiling);
    NsSiluGrad::KernelSiluGrad<DTYPE_DY> op;
    op.Init(dy, x, dx, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.tileDataNum, tilingData.tailBlockNum, schMode);
    op.Process();
}
