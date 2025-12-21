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
 * \file add_rms_norm_quant_v2.cpp
 * \brief
 */
#include "../add_rms_norm_quant/add_rms_norm_quant.h"
#include "../add_rms_norm_quant/add_rms_norm_quant_split_d.h"
#include "../add_rms_norm_quant/add_rms_norm_quant_single_n.h"
using namespace AscendC;

#define INIT_AND_PROCESS                                                                                             \
    do {                                                                                                             \
        op.Init(x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, bias, y1, y2, x, res_out, &tilingData); \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_quant_v2(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR bias, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) {
        KernelAddRmsNormQuant<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, false> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(10010)) {
        KernelAddRmsNormQuantSplitD<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, false> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(30010)) {
        KernelAddRmsNormQuantSingleN<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, false> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(101)) {
        KernelAddRmsNormQuant<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, true, false, true> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(10101)) {
        KernelAddRmsNormQuantSplitD<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, true, false, true> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(30101)) {
        KernelAddRmsNormQuantSingleN<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, true, false, true> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1011)) {
        KernelAddRmsNormQuant<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, true> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(11011)) {
        KernelAddRmsNormQuantSplitD<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, true> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(31011)) {
        KernelAddRmsNormQuantSingleN<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, false, true, true> op(&pipe);
        INIT_AND_PROCESS;
    }
}