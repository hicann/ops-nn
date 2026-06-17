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
 * \file softmax_grad_ext.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/softmax_grad_ext_ar_full_load.h"
#include "arch35/softmax_grad_ext_ar_small_r.h"
#include "arch35/softmax_grad_ext_ar_recompute.h"

using namespace SoftmaxGradExt;

namespace {
#define TILINGKEY_AR_SMALL_R 500
#define TILINGKEY_AR 1000
#define TILINGKEY_AR_RECOMPUTE 2000
} // namespace

extern "C" __global__ __aicore__ void softmax_grad_ext(
    GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILINGKEY_AR_SMALL_R)) {
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradExtARSmallRTilingData, tilingDataIn, tiling);
        const SoftmaxGradExtARSmallRTilingData* __restrict tilingData = &tilingDataIn;
        TPipe pipe;
        SoftmaxGradExtARSmallR<DTYPE_GRAD> op(&pipe);
        op.Init(grad, x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_AR)) {
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradExtARTilingData, tilingDataIn, tiling);
        const SoftmaxGradExtARTilingData* __restrict tilingData = &tilingDataIn;
        TPipe pipe;
        SoftmaxGradExtAR<DTYPE_GRAD> op(&pipe);
        op.Init(grad, x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_AR_RECOMPUTE)) {
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradExtARRecomputeTilingData, tilingDataIn, tiling);
        const SoftmaxGradExtARRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        TPipe pipe;
        SoftmaxGradExtARRecompute<DTYPE_GRAD> op(&pipe);
        op.Init(grad, x1, x2, y, tilingData);
        op.Process();
    }
}