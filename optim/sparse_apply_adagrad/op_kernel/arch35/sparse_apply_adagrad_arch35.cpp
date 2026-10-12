/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_apply_adagrad_arch35.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_apply_adagrad_tiling_data.h"
#include "sparse_apply_adagrad_small.h"
#include "sparse_apply_adagrad_big.h"
#include "sparse_apply_adagrad_optimize.h"
#include "sparse_apply_adagrad_optimize_split_r.h"

using namespace AscendC;
using namespace SparseApplyAdagradOps;

#define TILING_KEY_BIG 10002
#define TILING_KEY_SMALL 10001
#define TILINGKEY_OPTIMIZE 10000
#define TILINGKEY_OPTIMIZE_SPLIT_R 10010

#define SPARSE_APPLY_ADAGRAD_OPTIMIZE_IMPL()                                                            \
    do {                                                                                                \
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10000", SparseApplyAdagradOptimizeTilingData); \
        GET_TILING_DATA_WITH_STRUCT(SparseApplyAdagradOptimizeTilingData, tilingDataIn, tiling);        \
        const SparseApplyAdagradOptimizeTilingData* __restrict tilingData = &tilingDataIn;              \
        TPipe pipe;                                                                                     \
        SparseApplyAdagradOptimize op(&pipe);                                                           \
        op.Init(var, accum, lr, grad, indices, tilingData);                                             \
        op.Process();                                                                                   \
    } while (0)

#define SPARSE_APPLY_ADAGRAD_OPTIMIZE_SPLIT_R_IMPL()                                                          \
    do {                                                                                                      \
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10010", SparseApplyAdagradOptimizeSplitRTilingData); \
        GET_TILING_DATA_WITH_STRUCT(SparseApplyAdagradOptimizeSplitRTilingData, tilingDataIn, tiling);        \
        const SparseApplyAdagradOptimizeSplitRTilingData* __restrict tilingData = &tilingDataIn;              \
        TPipe pipe;                                                                                           \
        SparseApplyAdagradOptimizeSplitR op(&pipe);                                                           \
        op.Init(var, accum, lr, grad, indices, tilingData);                                                   \
        op.Process();                                                                                         \
    } while (0)

#define SPARSE_APPLY_ADAGRAD_SMALL_IMPL()                                                               \
    do {                                                                                                \
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10001", SparseApplyAdagradSmallDimTilingData); \
        GET_TILING_DATA_WITH_STRUCT(SparseApplyAdagradSmallDimTilingData, tilingDataIn, tiling);        \
        const SparseApplyAdagradSmallDimTilingData* __restrict tilingData = &tilingDataIn;              \
        TPipe pipe;                                                                                     \
        SparseApplyAdagradSmall op(&pipe);                                                              \
        op.Init(var, accum, lr, grad, indices, tilingData);                                             \
        op.Process();                                                                                   \
    } while (0)

#define SPARSE_APPLY_ADAGRAD_BIG_IMPL()                                                               \
    do {                                                                                              \
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 10002", SparseApplyAdagradBigDimTilingData); \
        GET_TILING_DATA_WITH_STRUCT(SparseApplyAdagradBigDimTilingData, tilingDataIn, tiling);        \
        const SparseApplyAdagradBigDimTilingData* __restrict tilingData = &tilingDataIn;              \
        TPipe pipe;                                                                                   \
        SparseApplyAdagradBig op(&pipe);                                                              \
        op.Init(var, accum, lr, grad, indices, tilingData);                                           \
        op.Process();                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void sparse_apply_adagrad(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
                                                           GM_ADDR indices, GM_ADDR var_ref, GM_ADDR accum_ref,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SparseApplyAdagradOptimizeTilingData);
    if (TILING_KEY_IS(TILINGKEY_OPTIMIZE)) {
        SPARSE_APPLY_ADAGRAD_OPTIMIZE_IMPL();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL)) {
        SPARSE_APPLY_ADAGRAD_SMALL_IMPL();
    } else if (TILING_KEY_IS(TILING_KEY_BIG)) {
        SPARSE_APPLY_ADAGRAD_BIG_IMPL();
    } else if (TILING_KEY_IS(TILINGKEY_OPTIMIZE_SPLIT_R)) {
        SPARSE_APPLY_ADAGRAD_OPTIMIZE_SPLIT_R_IMPL();
    }
}
