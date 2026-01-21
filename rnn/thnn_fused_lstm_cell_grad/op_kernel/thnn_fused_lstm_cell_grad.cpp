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
 * \file thnn_fused_lstm_cell_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "thnn_fused_lstm_cell_grad.h"
using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)                  \
    do {                                                     \
        templateClass<__VA_ARGS__> op;                       \
        op.Init(dhy, dc, cx, cy, storage, dgates, dc_prev,          \
                db, &tiling_data,                       \
                workspace, &pipe);                           \
        op.Process();                                        \
    } while (0)

extern "C" __global__ __aicore__ void thnn_fused_lstm_cell_grad(GM_ADDR dhy, GM_ADDR dc, GM_ADDR cx, GM_ADDR cy,
    GM_ADDR storage, GM_ADDR dgates, GM_ADDR dc_prev, GM_ADDR db, GM_ADDR workspace, GM_ADDR lstmCellGradTiling)
{
    GET_TILING_DATA(tiling_data, lstmCellGradTiling);
    const ThnnFusedLstmCellGradTilingData* __restrict tilingData = &tiling_data;
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        GENERAL_OP_IMPL(ThnnFusedLstmCellGrad, DTYPE_DHY);
    }
}