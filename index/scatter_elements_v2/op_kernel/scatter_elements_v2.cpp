/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#include "scatter_elements_v2_310p.h"
#else
#include "scatter_elements_v2.h"
#endif

#define CALL_OP_IMPL(...)                                    \
    do {                                                     \
        KernelScatterElementsV2<__VA_ARGS__> op;             \
        op.Init(tilingDevice, &pipe, var, indices, updates); \
        if (tilingDevice->modeFlag == 1) {                   \
            op.ProcessSmall();                               \
        } else {                                             \
            op.ProcessScatter();                             \
        }                                                    \
    } while (0)

extern "C" __global__ __aicore__ void scatter_elements_v2(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterElementsV2TilingData* __restrict tilingDevice = &tiling_data;
    AscendC::TPipe pipe;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#if (defined(DTYPE_VAR))
    if (TILING_KEY_IS(0)) {
        ScatterElementsV2310PNS::scatter_elements_v2_310p_split_m<DTYPE_VAR, DTYPE_INDICES>(var, indices, updates, &tiling_data, &pipe);
    } else if (TILING_KEY_IS(1)) {
        ScatterElementsV2310PNS::scatter_elements_v2_310p_multy_m<DTYPE_VAR, DTYPE_INDICES>(var, indices, updates, &tiling_data, &pipe);
    }
#endif
#else
    if (TILING_KEY_IS(111)) {
        CALL_OP_IMPL(float, int, 1);
    } else if (TILING_KEY_IS(112)) {
        CALL_OP_IMPL(float, int, 2);
    } else if (TILING_KEY_IS(121)) {
        CALL_OP_IMPL(float, long, 1);
    } else if (TILING_KEY_IS(122)) {
        CALL_OP_IMPL(float, long, 2);
    } else if (TILING_KEY_IS(211)) {
        CALL_OP_IMPL(half, int, 1);
    } else if (TILING_KEY_IS(212)) {
        CALL_OP_IMPL(half, int, 2);
    } else if (TILING_KEY_IS(221)) {
        CALL_OP_IMPL(half, long, 1);
    } else if (TILING_KEY_IS(222)) {
        CALL_OP_IMPL(half, long, 2);
    } else if (TILING_KEY_IS(311)) {
        CALL_OP_IMPL(int, int, 1);
    } else if (TILING_KEY_IS(312)) {
        CALL_OP_IMPL(int, int, 2);
    } else if (TILING_KEY_IS(321)) {
        CALL_OP_IMPL(int, long, 1);
    } else if (TILING_KEY_IS(322)) {
        CALL_OP_IMPL(int, long, 2);
    } else if (TILING_KEY_IS(411)) {
        CALL_OP_IMPL(uint8_t, int, 1);
    } else if (TILING_KEY_IS(412)) {
        CALL_OP_IMPL(uint8_t, int, 2);
    } else if (TILING_KEY_IS(421)) {
        CALL_OP_IMPL(uint8_t, long, 1);
    } else if (TILING_KEY_IS(422)) {
        CALL_OP_IMPL(uint8_t, long, 2);
    } else if (TILING_KEY_IS(511)) {
        CALL_OP_IMPL(int8_t, int, 1);
    } else if (TILING_KEY_IS(512)) {
        CALL_OP_IMPL(int8_t, int, 2);
    } else if (TILING_KEY_IS(521)) {
        CALL_OP_IMPL(int8_t, long, 1);
    } else if (TILING_KEY_IS(522)) {
        CALL_OP_IMPL(int8_t, long, 2);
    } else if (TILING_KEY_IS(611)) {
        CALL_OP_IMPL(bfloat16_t, int, 1);
    } else if (TILING_KEY_IS(612)) {
        CALL_OP_IMPL(bfloat16_t, int, 2);
    } else if (TILING_KEY_IS(621)) {
        CALL_OP_IMPL(bfloat16_t, long, 1);
    } else if (TILING_KEY_IS(622)) {
        CALL_OP_IMPL(bfloat16_t, long, 2);
    }
#endif
}
