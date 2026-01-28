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
 * \file softmax_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/softmax_grad_ar_full_load.h"
#include "arch35/softmax_grad_ar_small_r.h"
#include "arch35/softmax_grad_ar_recompute.h"
#include "arch35/softmax_grad_ara_full_load.h"
#include "arch35/softmax_grad_ara_recompute.h"

using namespace AscendC;
using namespace SoftmaxGradOps;

namespace
{
#define TILINGKEY_AR_SMALL_R 500
#define TILINGKEY_AR 1000
#define TILINGKEY_AR_RECOMPUTE 2000
#define TILINGKEY_ARA 10000
#define TILINGKEY_ARA_RECOMPUTE 20000

}  // namespace

#define softmax_grad_AR_SMALL_R_IMPL(INPUT_TYPE)                                            \
    do {                                                                            \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARSmallRTilingData, tilingDataIn, tiling); \
        const SoftmaxGradARSmallRTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                                 \
        SoftmaxGradARSmallR<INPUT_TYPE> op(&pipe);                                        \
        op.Init(softmax, grad_softmax, grad_x, tilingData);                         \
        op.Process();                                                               \
    } while (0)

#define softmax_grad_AR_IMPL(INPUT_TYPE)                                            \
    do {                                                                            \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARTilingData, tilingDataIn, tiling); \
        const SoftmaxGradARTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                                 \
        SoftmaxGradAR<INPUT_TYPE> op(&pipe);                                        \
        op.Init(softmax, grad_softmax, grad_x, tilingData);                         \
        op.Process();                                                               \
    } while (0)

#define softmax_grad_AR_RECOMPUTE_IMPL(INPUT_TYPE)                                           \
    do {                                                                                     \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARRecomputeTilingData, tilingDataIn, tiling); \
        const SoftmaxGradARRecomputeTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                                          \
        SoftmaxGradArRecompute<INPUT_TYPE> op(&pipe);                                        \
        op.Init(softmax, grad_softmax, grad_x, tilingData);                                  \
        op.Process();                                                                        \
    } while (0)

#define softmax_grad_ARA_IMPL(INPUT_TYPE)                                              \
    do {                                                                               \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARATilingData, tiling_data_in, tiling); \
        const SoftmaxGradARATilingData* __restrict tilingData = &tiling_data_in;       \
        TPipe pipe;                                                                    \
        SoftmaxGradARA<INPUT_TYPE> op(tilingData);                                     \
        op.Init(softmax, grad_softmax, grad_x, &pipe);                                 \
        op.Process();                                                                  \
    } while (0)

#define softmax_grad_ARA_RECOMPUTE_IMPL(INPUT_TYPE)                                             \
    do {                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxGradARARecomputeTilingData, tiling_data_in, tiling); \
        const SoftmaxGradARARecomputeTilingData* __restrict tilingData = &tiling_data_in;       \
        TPipe pipe;                                                                             \
        SoftmaxGradARARecompute<INPUT_TYPE> op(tilingData);                                     \
        op.Init(softmax, grad_softmax, grad_x, &pipe);                                          \
        op.Process();                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void softmax_grad(GM_ADDR softmax, GM_ADDR grad_softmax, GM_ADDR grad_x,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILINGKEY_AR_SMALL_R)) {
        softmax_grad_AR_SMALL_R_IMPL(DTYPE_SOFTMAX);
    } else if (TILING_KEY_IS(TILINGKEY_AR)) {
        softmax_grad_AR_IMPL(DTYPE_SOFTMAX);
    } else if (TILING_KEY_IS(TILINGKEY_AR_RECOMPUTE)) {
        softmax_grad_AR_RECOMPUTE_IMPL(DTYPE_SOFTMAX);
    } else if (TILING_KEY_IS(TILINGKEY_ARA)) {
        softmax_grad_ARA_IMPL(DTYPE_SOFTMAX);
    } else if (TILING_KEY_IS(TILINGKEY_ARA_RECOMPUTE)) {
        softmax_grad_ARA_RECOMPUTE_IMPL(DTYPE_SOFTMAX);
    }
}