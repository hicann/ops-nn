/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_apt.h
 * \brief
 */

#ifndef LAYER_NORM_V3_APT_H
#define LAYER_NORM_V3_APT_H

#include "kernel_operator.h"
#include "arch35/layer_norm_v3_two_pass.h"
#include "arch35/layer_norm_v3_welford.h"
#include "arch35/layer_norm_v3_two_pass_perf.h"
#include "arch35/layer_norm_v3_no_reduce.h"

using namespace LayerNormV3;
using AscendC::AIC;

#define LNV3_REGBASE_NO_REDUCE_FLOAT_FLOAT 600
#define LNV3_REGBASE_NO_REDUCE_HALF_FLOAT 610
#define LNV3_REGBASE_NO_REDUCE_HALF_HALF 611
#define LNV3_REGBASE_NO_REDUCE_BF16_FLOAT 620
#define LNV3_REGBASE_NO_REDUCE_BF16_BF16 622

#define LNV3_REGBASE_TWO_PASS_FLOAT_FLOAT 300
#define LNV3_REGBASE_TWO_PASS_HALF_FLOAT 310
#define LNV3_REGBASE_TWO_PASS_HALF_HALF 311
#define LNV3_REGBASE_TWO_PASS_BF16_FLOAT 320
#define LNV3_REGBASE_TWO_PASS_BF16_BF16 322

#define LNV3_REGBASE_WELFORD_FLOAT_FLOAT 400
#define LNV3_REGBASE_WELFORD_HALF_FLOAT 410
#define LNV3_REGBASE_WELFORD_HALF_HALF 411
#define LNV3_REGBASE_WELFORD_BF16_FLOAT 420
#define LNV3_REGBASE_WELFORD_BF16_BF16 422

#define LNV3_REGBASE_TWO_PASS_PERF_FLOAT_FLOAT 500
#define LNV3_REGBASE_TWO_PASS_PERF_HALF_FLOAT 510
#define LNV3_REGBASE_TWO_PASS_PERF_HALF_HALF 511
#define LNV3_REGBASE_TWO_PASS_PERF_BF16_FLOAT 520
#define LNV3_REGBASE_TWO_PASS_PERF_BF16_BF16 522

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseNoReduceImpl(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseNoReduce, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseNoReduce* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseNoReduce<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseTwoPassImpl(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseTwoPass, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseTwoPass* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseTwoPass<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseWelfordImpl(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataWelford, tiling_data_in, tiling);
    const LayerNormV3TilingDataWelford* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseWelford<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean>
__aicore__ inline void RegbaseTwoPassPerfImpl(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseTwoPassPerf, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseTwoPassPerf* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseTwoPassPerf<Tfm, Tweight, Tmean> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, rstd);
    op.Process();
}

template <bool IsOutRstd>
__aicore__ inline void layer_norm_impl(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_FLOAT_FLOAT)) {
        RegbaseTwoPassImpl<float, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_HALF_FLOAT)) {
        RegbaseTwoPassImpl<half, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_HALF_HALF)) {
        RegbaseTwoPassImpl<half, half, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_BF16_FLOAT)) {
        RegbaseTwoPassImpl<bfloat16_t, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_BF16_BF16)) {
        RegbaseTwoPassImpl<bfloat16_t, bfloat16_t, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_FLOAT_FLOAT)) {
        RegbaseWelfordImpl<float, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_HALF_FLOAT)) {
        RegbaseWelfordImpl<half, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_HALF_HALF)) {
        RegbaseWelfordImpl<half, half, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_BF16_FLOAT)) {
        RegbaseWelfordImpl<bfloat16_t, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_BF16_BF16)) {
        RegbaseWelfordImpl<bfloat16_t, bfloat16_t, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF_FLOAT_FLOAT)) {
        RegbaseTwoPassPerfImpl<float, float, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF_HALF_FLOAT)) {
        RegbaseTwoPassPerfImpl<half, float, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF_HALF_HALF)) {
        RegbaseTwoPassPerfImpl<half, half, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF_BF16_FLOAT)) {
        RegbaseTwoPassPerfImpl<bfloat16_t, float, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF_BF16_BF16)) {
        RegbaseTwoPassPerfImpl<bfloat16_t, bfloat16_t, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE_FLOAT_FLOAT)) {
        RegbaseNoReduceImpl<float, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE_HALF_FLOAT)) {
        RegbaseNoReduceImpl<half, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE_HALF_HALF)) {
        RegbaseNoReduceImpl<half, half, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE_BF16_FLOAT)) {
        RegbaseNoReduceImpl<bfloat16_t, float, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE_BF16_BF16)) {
        RegbaseNoReduceImpl<bfloat16_t, bfloat16_t, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    }

    return;
}

#endif // LAYER_NORM_V3_APT_H