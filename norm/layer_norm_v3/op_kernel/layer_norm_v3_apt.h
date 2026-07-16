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
#include "arch35/layer_norm_v3_norm_not_equal_params.h"
#include "arch35/layer_norm_v3_welford_multi_reduce.h"
#include "arch35/layer_norm_v3_welford_multi_params.h"

using namespace LayerNormV3;
using AscendC::AIC;

#define LNV3_REGBASE_TWO_PASS 300
#define LNV3_REGBASE_WELFORD 400
#define LNV3_REGBASE_TWO_PASS_PERF 500
#define LNV3_REGBASE_NO_REDUCE 600
#define LNV3_REGBASE_NORM_NOT_EQUAL_PARAMS 700
#define LNV3_REGBASE_WELFORD_MULTI_REDUCE 800
#define LNV3_REGBASE_WELFORD_MULTI_PARAMS 900

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseNoReduceImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                           GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseNoReduce, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseNoReduce* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseNoReduce<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseTwoPassImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                          GM_ADDR lastout, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseTwoPass, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseTwoPass* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseTwoPass<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseWelfordImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                          GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataWelford, tiling_data_in, tiling);
    const LayerNormV3TilingDataWelford* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseWelford<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean>
__aicore__ inline void RegbaseTwoPassPerfImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                              GM_ADDR rstd, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseTwoPassPerf, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseTwoPassPerf* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseTwoPassPerf<Tfm, Tweight, Tmean> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, rstd);
    op.Process();
}
template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseNormNotEqualParamsImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                     GM_ADDR rstd, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataRegBaseNormNotEqualParams, tiling_data_in, tiling);
    const LayerNormV3TilingDataRegBaseNormNotEqualParams* __restrict tilingData = &tiling_data_in;
    LayerNormV3RegbaseNormNotEqualParams<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, rstd);
    op.Process();
}
template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseWelfordMultiReduceImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                     GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataWelfordMultiReduce, tiling_data_in, tiling);
    const LayerNormV3TilingDataWelfordMultiReduce* __restrict tilingData = &tiling_data_in;
    LayerNormV3WelfordMultiReduce<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <typename Tfm, typename Tweight, typename Tmean, bool IsOutRstd>
__aicore__ inline void RegbaseWelfordMultiParamsImpl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                     GM_ADDR lastout, GM_ADDR tiling)
{
    TPipe pipeIn;
    GET_TILING_DATA_WITH_STRUCT(LayerNormV3TilingDataWelfordMultiParams, tiling_data_in, tiling);
    const LayerNormV3TilingDataWelfordMultiParams* __restrict tilingData = &tiling_data_in;
    LayerNormV3WelfordMultiParams<Tfm, Tweight, Tmean, IsOutRstd> op(tilingData, &pipeIn);
    op.Init(x, gamma, beta, y, mean, lastout);
    op.Process();
}

template <bool IsOutRstd>
__aicore__ inline void layer_norm_impl(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout,
                                       GM_ADDR workspace, GM_ADDR tiling)
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
    if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS)) {
        RegbaseTwoPassImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD)) {
        RegbaseWelfordImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_PERF)) {
        RegbaseTwoPassPerfImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NO_REDUCE)) {
        RegbaseNoReduceImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout, tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_NORM_NOT_EQUAL_PARAMS)) {
        RegbaseNormNotEqualParamsImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout,
                                                                                   tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_MULTI_REDUCE)) {
        RegbaseWelfordMultiReduceImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout,
                                                                                   tiling);
        return;
    } else if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_MULTI_PARAMS)) {
        RegbaseWelfordMultiParamsImpl<DTYPE_X, DTYPE_GAMMA, DTYPE_MEAN, IsOutRstd>(x, gamma, beta, y, mean, lastout,
                                                                                   tiling);
        return;
    }

    return;
}

#endif // LAYER_NORM_V3_APT_H
