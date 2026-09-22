/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2.cpp
 * \brief
 */
#include "scatter_elements_v2.h"

template <typename T, typename U>
__aicore__ inline void ExecLegacyScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                           ScatterElementsV2TilingData* tiling_data, AscendC::TPipe* pipe)
{
    KernelScatterElementsV2<T, U> op;
    op.Init(tiling_data, pipe, var, indices, updates);
    if (tiling_data->mode == ScatterElementsV2NS::SCATTER_MODE_NONE) {
        if (tiling_data->modeFlag == SMALL_MODE) {
            op.ProcessNoneSmall();
        } else if (tiling_data->M != 0) {
            op.ProcessNoneStableBucket();
        } else {
            op.ProcessNoneScatter();
        }
    } else if (tiling_data->modeFlag == SMALL_MODE) {
        op.ProcessSmall();
    } else {
        op.ProcessScatter();
    }
}

template <typename T, typename U>
__aicore__ inline void ExecScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                     ScatterElementsV2TilingData* tiling_data, AscendC::TPipe* pipe)
{
    ExecLegacyScatterOp<T, U>(var, indices, updates, tiling_data, pipe);
}
#define CALL_OP_IMPL(T, U)                                               \
    do {                                                                 \
        ExecScatterOp<T, U>(var, indices, updates, tilingDevice, &pipe); \
    } while (0)

extern "C" __global__ __aicore__ void scatter_elements_v2(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    ScatterElementsV2TilingData* __restrict tilingDevice = &tiling_data;
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(110)) {
        CALL_OP_IMPL(float, int);
    } else if (TILING_KEY_IS(120)) {
        CALL_OP_IMPL(float, long);
    } else if (TILING_KEY_IS(210)) {
        CALL_OP_IMPL(half, int);
    } else if (TILING_KEY_IS(220)) {
        CALL_OP_IMPL(half, long);
    } else if (TILING_KEY_IS(310)) {
        CALL_OP_IMPL(int, int);
    } else if (TILING_KEY_IS(320)) {
        CALL_OP_IMPL(int, long);
    } else if (TILING_KEY_IS(410)) {
        CALL_OP_IMPL(uint8_t, int);
    } else if (TILING_KEY_IS(420)) {
        CALL_OP_IMPL(uint8_t, long);
    } else if (TILING_KEY_IS(510)) {
        CALL_OP_IMPL(int8_t, int);
    } else if (TILING_KEY_IS(520)) {
        CALL_OP_IMPL(int8_t, long);
    } else if (TILING_KEY_IS(610)) {
        CALL_OP_IMPL(bfloat16_t, int);
    } else if (TILING_KEY_IS(620)) {
        CALL_OP_IMPL(bfloat16_t, long);
    } else if (TILING_KEY_IS(710)) {
        CALL_OP_IMPL(int16_t, int);
    } else if (TILING_KEY_IS(720)) {
        CALL_OP_IMPL(int16_t, long);
    } else if (TILING_KEY_IS(810)) {
        CALL_OP_IMPL(int64_t, int);
    } else if (TILING_KEY_IS(820)) {
        CALL_OP_IMPL(int64_t, long);
    } else if (TILING_KEY_IS(910)) {
        CALL_OP_IMPL(double, int);
    } else if (TILING_KEY_IS(920)) {
        CALL_OP_IMPL(double, long);
    }
}
