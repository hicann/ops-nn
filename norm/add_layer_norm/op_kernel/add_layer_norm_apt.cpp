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
 * \file add_layer_norm_apt.cpp
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/add_layer_norm_regbase_full_load.h"
#include "arch35/add_layer_norm_regbase_welford.h"

#define TILING_FULL_LOAD_BIAS_NONE 8000
#define TILING_FULL_LOAD_BIAS_ELEWISE 8001
#define TILING_FULL_LOAD_BIAS_BRC 8002
#define TILING_WELFORD_BIAS_NONE 8100
#define TILING_WELFORD_BIAS_ELEWISE 8101
#define TILING_WELFORD_BIAS_BRC 8102

extern "C" __global__ __aicore__ void add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(AddLayerNormRegbaseTilingData, tilingDataIn, tiling);
    const AddLayerNormRegbaseTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_NONE)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_FULL_LOAD_BIAS_NONE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE)) {
        AddLayerNorm::RegbaseFullLoad<
            DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_FULL_LOAD_BIAS_ELEWISE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_FULL_LOAD_BIAS_BRC>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_NONE)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_WELFORD_BIAS_NONE> op(
            tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_WELFORD_BIAS_ELEWISE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, DTYPE_Y, TILING_WELFORD_BIAS_BRC> op(
            tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    }
}