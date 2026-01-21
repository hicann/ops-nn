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
 * \file single_layer_lstm_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "single_layer_lstm_grad.h"
#include "matmul_config.h"
using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        templateClass<__VA_ARGS__> op;                       \
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(),       \
                          op.dwMM, dwMMTiling,               \
                          op.dgateMM, dgateMMTiling);        \
        op.Init(x, w, b, y, init_h, init_c, h, c, dy, dh,    \
                dc, i, j, f, o, tanhct, seq_length, dw,      \
                db, dx, dh_prev, dc_prev, &tiling_data,      \
                workspace, &pipe);                           \
        op.Process();                                        \
    } while (0)

extern "C" __global__ __aicore__ void single_layer_lstm_grad(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR init_h,
    GM_ADDR init_c, GM_ADDR h, GM_ADDR c, GM_ADDR dy, GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f,
    GM_ADDR o, GM_ADDR tanhct, GM_ADDR seq_length, GM_ADDR dw, GM_ADDR db, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dc_prev,
    GM_ADDR workspace, GM_ADDR rnnGradTiling)
{
    GET_TILING_DATA(tiling_data, rnnGradTiling);
    const SingleLayerLstmGradTilingData* __restrict tilingData = &tiling_data;
    const TCubeTiling* __restrict dwMMTiling = &(tilingData->dwMMParam);
    const TCubeTiling* __restrict dgateMMTiling = &(tilingData->dgateMMParam);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, false, false);
    } else if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, false, false);
    } else if (TILING_KEY_IS(100)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, true, false);
    } else if (TILING_KEY_IS(101)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, true, false);
    } else if (TILING_KEY_IS(110)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, true, false);
    } else if (TILING_KEY_IS(111)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, true, false);
    } else if (TILING_KEY_IS(1000)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, false, true);
    } else if (TILING_KEY_IS(1001)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, false, true);
    } else if (TILING_KEY_IS(1010)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, false, true);
    } else if (TILING_KEY_IS(1011)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, false, true);
    } else if (TILING_KEY_IS(1100)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1101)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_CFG, true, true);
    } else if (TILING_KEY_IS(1110)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_CFG, MM_HUGE_CFG, true, true);
    } else if (TILING_KEY_IS(1111)) {
        GENERAL_OP_IMPL(RNNGrad, DTYPE_X, MM_HUGE_CFG, MM_HUGE_CFG, true, true);
    }

    #ifdef __CCE_KT_TEST__
    EmptyTestFunc();
    #endif  // __CCE_KT_TEST__
}