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
 * \file group_norm_silu_apt.cpp
 * \brief
 */

#include "arch35/group_norm_silu_regbase_empty_tensor.h"
#include "arch35/group_norm_silu_regbase_welford.h"
#include "arch35/group_norm_silu_regbase_two_pass.h"
#include "arch35/group_norm_silu_regbase_two_pass_generalized.h"
#include "arch35/group_norm_silu_regbase_welford_generalized.h"

using namespace GroupNormSilu;

namespace {
#define TILINGKEY_EMPTY_TENSOR 1000
#define TILINGKEY_EMPTY_TENSOR_MIX_TYPE 1001
#define TILINGKEY_WELFORD_PERF 1100
#define TILINGKEY_TWOPASS_PERF 1110
#define TILINGKEY_WELFORD_GENERALIZED 1120
#define TILINGKEY_TWOPASS_GENERALIZED 1130
#define TILINGKEY_WELFORD_PERF_MIX_TYPE 1101
#define TILINGKEY_TWOPASS_PERF_MIX_TYPE 1111
#define TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE 1121
#define TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE 1131
} // namespace

extern "C" __global__ __aicore__ void group_norm_silu(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(GroupNormSiluRegbaseTilingData, tilingDataIn, tiling);
    const GroupNormSiluRegbaseTilingData* __restrict tilingData = &tilingDataIn;
    if (TILING_KEY_IS(TILINGKEY_WELFORD_PERF)) {
        GroupNormSilu::GroupNormSiluWelford<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_PERF_MIX_TYPE)) {
        GroupNormSilu::GroupNormSiluWelford<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_PERF)) {
        GroupNormSilu::GroupNormSiluTwoPass<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_PERF_MIX_TYPE)) {
        GroupNormSilu::GroupNormSiluTwoPass<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_GENERALIZED)) {
        GroupNormSilu::GroupNormSiluWelfordGeneralized<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE)) {
        GroupNormSilu::GroupNormSiluWelfordGeneralized<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_GENERALIZED)) {
        GroupNormSilu::GroupNormSiluTwoPassGeneralized<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE)) {
        GroupNormSilu::GroupNormSiluTwoPassGeneralized<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_EMPTY_TENSOR)) {
        GroupNormSilu::GroupNormSiluEmpty<DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_EMPTY_TENSOR_MIX_TYPE)) {
        GroupNormSilu::GroupNormSiluEmpty<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, tilingData);
        op.Process();
    }
}