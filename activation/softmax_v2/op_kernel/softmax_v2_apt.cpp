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
 * \file softmax_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/softmax_v2_ar_small_r.h"
#include "arch35/softmax_v2_ar_full_load.h"
#include "arch35/softmax_v2_ar_recompute.h"
#include "arch35/softmax_v2_ara_full_load.h"
#include "arch35/softmax_v2_ara_recompute.h"

using namespace AscendC;
using namespace SoftmaxV2Ops;

namespace
{
#define TILINGKEY_AR_SMALL_R 500
#define TILINGKEY_AR 1000
#define TILINGKEY_AR_RECOMPUTE 2000
#define TILINGKEY_ARA 10000
#define TILINGKEY_ARA_RECOMPUTE 20000

}  // namespace

#define SOFTMAX_V2_AR_SMALL_R_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                 \
    do {                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxV2ArSmallRTilingData, tilingDataIn, tiling);     \
        const SoftmaxV2ArSmallRTilingData* __restrict tilingData = &tilingDataIn;           \
        TPipe pipe;                                                                         \
        SoftmaxV2ArSmallR<INPUT_TYPE, OUTPUT_TYPE> op(&pipe);                               \
        op.Init(x, y, tilingData);                                                          \
        op.Process();                                                                       \
    } while (0)

#define SOFTMAX_V2_AR_IMPL(INPUT_TYPE, OUTPUT_TYPE)                               \
    do {                                                                          \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxV2ARTilingData, tilingDataIn, tiling); \
        const SoftmaxV2ARTilingData* __restrict tilingData = &tilingDataIn;       \
        TPipe pipe;                                                               \
        SoftmaxV2AR<INPUT_TYPE, OUTPUT_TYPE> op(&pipe);                           \
        op.Init(x, y, tilingData);                                                \
        op.Process();                                                             \
    } while (0)

#define SOFTMAX_V2_AR_RECOMPUTE_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                   \
    do {                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxV2ArRecomputeTilingData, tilingDataIn, tiling);      \
        const SoftmaxV2ArRecomputeTilingData* __restrict tilingData = &tilingDataIn;            \
        TPipe pipe;                                                                             \
        SoftmaxV2ArRecompute<INPUT_TYPE, OUTPUT_TYPE> op(&pipe);                                \
        op.Init(x, y, tilingData);                                                              \
        op.Process();                                                                           \
    } while (0)

#define SOFTMAX_V2_ARA_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                 \
    do {                                                                             \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxV2ARATilingData, tiling_data_in, tiling); \
        const SoftmaxV2ARATilingData* __restrict tilingData = &tiling_data_in;       \
        TPipe pipe;                                                                  \
        SoftmaxV2ARA<INPUT_TYPE, OUTPUT_TYPE> op(tilingData);                        \
        op.Init(x, y, &pipe);                                                        \
        op.Process();                                                                \
    } while (0)

#define SOFTMAX_V2_ARA_RECOMPUTE_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                \
    do {                                                                                      \
        GET_TILING_DATA_WITH_STRUCT(SoftmaxV2ARARecomputeTilingData, tiling_data_in, tiling); \
        const SoftmaxV2ARARecomputeTilingData* __restrict tilingData = &tiling_data_in;       \
        TPipe pipe;                                                                           \
        SoftmaxV2ARARecompute<INPUT_TYPE, OUTPUT_TYPE> op(tilingData);                        \
        op.Init(x, y, &pipe);                                                                 \
        op.Process();                                                                         \
    } while (0)

extern "C" __global__ __aicore__ void softmax_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILINGKEY_AR_SMALL_R)) {
        SOFTMAX_V2_AR_SMALL_R_IMPL(DTYPE_X, DTYPE_Y);
    } else if (TILING_KEY_IS(TILINGKEY_AR)) {
        SOFTMAX_V2_AR_IMPL(DTYPE_X, DTYPE_Y);
    } else if (TILING_KEY_IS(TILINGKEY_AR_RECOMPUTE)) {
        SOFTMAX_V2_AR_RECOMPUTE_IMPL(DTYPE_X, DTYPE_Y);
    } else if (TILING_KEY_IS(TILINGKEY_ARA)) {
        SOFTMAX_V2_ARA_IMPL(DTYPE_X, DTYPE_Y);
    } else if (TILING_KEY_IS(TILINGKEY_ARA_RECOMPUTE)) {
        SOFTMAX_V2_ARA_RECOMPUTE_IMPL(DTYPE_X, DTYPE_Y);
    }
}