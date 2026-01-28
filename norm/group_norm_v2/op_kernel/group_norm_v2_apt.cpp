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
 * \file group_norm_v2_apt.cpp
 * \brief
 */

#include "arch35/group_norm_v2_regbase_welford.h"
#include "arch35/group_norm_v2_regbase_two_pass.h"
#include "arch35/group_norm_v2_regbase_two_pass_generalized.h"
#include "arch35/group_norm_v2_regbase_welford_generalized.h"
using namespace GroupNormV2;

namespace {
#define TILINGKEY_WELFORD_PERF 1100
#define TILINGKEY_TWOPASS_PERF 1110
#define TILINGKEY_WELFORD_GENERALIZED 1120
#define TILINGKEY_TWOPASS_GENERALIZED 1130
#define TILINGKEY_WELFORD_PERF_MIX_TYPE 1101
#define TILINGKEY_TWOPASS_PERF_MIX_TYPE 1111
#define TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE 1121
#define TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE 1131
}  // namespace

extern "C" __global__ __aicore__ void group_norm_v2(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                    GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (TILING_KEY_IS(TILINGKEY_WELFORD_PERF)) {
    GroupNormV2::GroupNormV2Welford<DTYPE_X, DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_WELFORD_PERF_MIX_TYPE)) {
    GroupNormV2::GroupNormV2Welford<DTYPE_X, float> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_PERF)) {
    GroupNormV2::GroupNormV2TwoPass<DTYPE_X, DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_PERF_MIX_TYPE)) {
    GroupNormV2::GroupNormV2TwoPass<DTYPE_X, float> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_WELFORD_GENERALIZED)) {
    GroupNormV2::GroupNormV2WelfordGeneralized<DTYPE_X, DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE)) {
    GroupNormV2::GroupNormV2WelfordGeneralized<DTYPE_X, float> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_GENERALIZED)) {
    GroupNormV2::GroupNormV2TwoPassGeneralized<DTYPE_X, DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE)) {
    GroupNormV2::GroupNormV2TwoPassGeneralized<DTYPE_X, float> op;
    op.Init(x, gamma, beta, y, mean, rstd, userWS, &tilingData);
    op.Process();
  }
}
