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
 * \file add_layer_norm.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_ARCH35_H
#define ADD_LAYER_NORM_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "add_layer_norm_regbase_full_load.h"
#include "add_layer_norm_regbase_welford.h"
#include "add_layer_norm_regbase_reduce_empty.h"

#define TILING_FULL_LOAD_BIAS_NONE 8000
#define TILING_FULL_LOAD_BIAS_ELEWISE 8001
#define TILING_FULL_LOAD_BIAS_BRC 8002
#define TILING_FULL_LOAD_NO_DB_BIAS_NONE 8010
#define TILING_FULL_LOAD_NO_DB_BIAS_ELEWISE 8011
#define TILING_FULL_LOAD_NO_DB_BIAS_BRC 8012
#define TILING_WELFORD_BIAS_NONE 8100
#define TILING_WELFORD_BIAS_ELEWISE 8101
#define TILING_WELFORD_BIAS_BRC 8102
#define TILING_REDUCE_EMPTY 8200

template <typename Y_TYPE>
__aicore__ inline void AddLayerNormImpl(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR y,
                                        GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(AddLayerNormRegbaseTilingData, tilingDataIn, tiling);
    const AddLayerNormRegbaseTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_NONE)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE, TILING_FULL_LOAD_BIAS_NONE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE,
                                      TILING_FULL_LOAD_BIAS_ELEWISE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE, TILING_FULL_LOAD_BIAS_BRC>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_NO_DB_BIAS_NONE)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE,
                                      TILING_FULL_LOAD_NO_DB_BIAS_NONE, AddLayerNorm::SINGLE_BUFFER_NUM>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_NO_DB_BIAS_ELEWISE)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE,
                                      TILING_FULL_LOAD_NO_DB_BIAS_ELEWISE, AddLayerNorm::SINGLE_BUFFER_NUM>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_NO_DB_BIAS_BRC)) {
        AddLayerNorm::RegbaseFullLoad<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE,
                                      TILING_FULL_LOAD_NO_DB_BIAS_BRC, AddLayerNorm::SINGLE_BUFFER_NUM>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_NONE)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE, TILING_WELFORD_BIAS_NONE> op(
            tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE, TILING_WELFORD_BIAS_ELEWISE>
            op(tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC)) {
        AddLayerNorm::RegbaseWelford<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_BETA, Y_TYPE, TILING_WELFORD_BIAS_BRC> op(
            tilingData);
        op.Init(x1, x2, gamma, beta, bias, y, mean, rstd, x);
        op.Process();
    } else if (TILING_KEY_IS(TILING_REDUCE_EMPTY)) {
        AddLayerNorm::RegbaseReduceEmpty op(tilingData);
        op.Init(mean, rstd);
        op.Process();
    }
}

#endif // ADD_LAYER_NORM_ARCH35_H
