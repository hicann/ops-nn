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
 * \file group_normalization_grad.cpp
 * \brief
 */

#include "group_normalization_grad_tiling_key.h"
#include "group_normalization_grad.h"
using namespace AscendC;

template <uint32_t schMode>
__global__ __aicore__ void group_normalization_grad(GM_ADDR x, GM_ADDR dy, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                                    GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupNormalizationGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupNormalizationGradTilingData, tilingData, tiling);

#ifdef __CCE_KT_TEST__
    // UT/CPU 仿真：schMode 0/1/2 区分 dtype，一份二进制测三种类型
    if constexpr (schMode == 0) {
        NsGroupNormalizationGrad::KernelGroupNormalizationGrad<float> op;
        op.Init(x, dy, gamma, mean, rstd, dx, tilingData.groupElemNum, tilingData.groupCount,
                tilingData.smallCoreGroupNum, tilingData.bigCoreGroupNum, tilingData.finalGroupTileNum,
                tilingData.tileDataNum, tilingData.alignedTileDataNum, tilingData.tailDataNum, tilingData.tailBlockNum,
                tilingData.groupElemNumFloat, tilingData.groupElemNumReciprocal);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsGroupNormalizationGrad::KernelGroupNormalizationGrad<half> op;
        op.Init(x, dy, gamma, mean, rstd, dx, tilingData.groupElemNum, tilingData.groupCount,
                tilingData.smallCoreGroupNum, tilingData.bigCoreGroupNum, tilingData.finalGroupTileNum,
                tilingData.tileDataNum, tilingData.alignedTileDataNum, tilingData.tailDataNum, tilingData.tailBlockNum,
                tilingData.groupElemNumFloat, tilingData.groupElemNumReciprocal);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsGroupNormalizationGrad::KernelGroupNormalizationGrad<bfloat16_t> op;
        op.Init(x, dy, gamma, mean, rstd, dx, tilingData.groupElemNum, tilingData.groupCount,
                tilingData.smallCoreGroupNum, tilingData.bigCoreGroupNum, tilingData.finalGroupTileNum,
                tilingData.tileDataNum, tilingData.alignedTileDataNum, tilingData.tailDataNum, tilingData.tailBlockNum,
                tilingData.groupElemNumFloat, tilingData.groupElemNumReciprocal);
        op.Process();
    }
#else
    // NPU：构建系统按 def DataType 注入 DTYPE_X 宏，编译出 3 个 .o（fp16/fp32/bf16）
    NsGroupNormalizationGrad::KernelGroupNormalizationGrad<DTYPE_X> op;
    op.Init(x, dy, gamma, mean, rstd, dx, tilingData.groupElemNum, tilingData.groupCount, tilingData.smallCoreGroupNum,
            tilingData.bigCoreGroupNum, tilingData.finalGroupTileNum, tilingData.tileDataNum,
            tilingData.alignedTileDataNum, tilingData.tailDataNum, tilingData.tailBlockNum,
            tilingData.groupElemNumFloat, tilingData.groupElemNumReciprocal);
    op.Process();
#endif
}
