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
 * \file confusion_softmax_grad_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/confusion_softmax_grad_ar_full_load.h"
#include "arch35/confusion_softmax_grad_ar_recompute.h"
#include "arch35/confusion_softmax_grad_ar_small_r.h"

using namespace AscendC;
using namespace ConfusionSoftmaxGradOps;

namespace
{
#define TILINGKEY_AR_SMALL_R 500
#define TILINGKEY_AR 1000
#define TILINGKEY_AR_RECOMPUTE 2000

}  // namespace

#define CONFUSION_SOFTMAX_GRAD_AR_SMALL_R_IMPL(INPUT_TYPE)                               \
    do {                                                                          \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARSmallRTilingData, tilingDataIn, tiling); \
        const SoftmaxGradARSmallRTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                               \
        ConfusionSoftmaxGradARSmallR<INPUT_TYPE> op(&pipe);                           \
        op.Init(grad, x, y, tilingData);                                                \
        op.Process();                                                             \
    } while (0)

#define CONFUSION_SOFTMAX_GRAD_AR_IMPL(INPUT_TYPE)                               \
    do {                                                                          \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARTilingData, tilingDataIn, tiling); \
        const SoftmaxGradARTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                               \
        ConfusionSoftmaxGradAR<INPUT_TYPE> op(&pipe);                           \
        op.Init(grad, x, y, tilingData);                                                \
        op.Process();                                                             \
    } while (0)

#define CONFUSION_SOFTMAX_GRAD_AR_RECOMPUTE_IMPL(INPUT_TYPE)                                   \
    do {                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARRecomputeTilingData, tilingDataIn, tiling);      \
        const SoftmaxGradARRecomputeTilingData* __restrict tilingData = &tilingDataIn;            \
        TPipe pipe;                                                                             \
        ConfusionSoftmaxGradArRecompute<INPUT_TYPE> op(&pipe);                                \
        op.Init(grad, x, y, tilingData);                                                              \
        op.Process();                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void confusion_softmax_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILINGKEY_AR_SMALL_R)) {
        CONFUSION_SOFTMAX_GRAD_AR_SMALL_R_IMPL(DTYPE_X);
    } else if (TILING_KEY_IS(TILINGKEY_AR)) {
        CONFUSION_SOFTMAX_GRAD_AR_IMPL(DTYPE_X);
    } else if (TILING_KEY_IS(TILINGKEY_AR_RECOMPUTE)) {
        CONFUSION_SOFTMAX_GRAD_AR_RECOMPUTE_IMPL(DTYPE_X);
    }
}