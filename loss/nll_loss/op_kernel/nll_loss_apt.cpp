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
 * \file nll_loss.cpp
 * \brief nll_loss
 */

#include "arch35/nll_loss_simt.h"
#include "arch35/nll_loss_mix.h"
#include "arch35/nll_loss_tiling_key.h"
using namespace AscendC;

template <uint64_t schId, uint64_t xDims, uint64_t reduction>
__global__ __aicore__ void nll_loss(
    GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR totalWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (schId == 0) {
        KernelNLLLossSimt<DTYPE_X, DTYPE_TARGET, schId, xDims, reduction> op;
        op.Init(x, target, weight, y, totalWeight, workspace, tilingData);
        op.Process();
    } else if constexpr (schId == 1) {
        KernelNLLLossSimd<DTYPE_X, DTYPE_TARGET> op;
        op.Init(x, target, weight, y, totalWeight, workspace, tilingData);
        op.Process();
    }
}