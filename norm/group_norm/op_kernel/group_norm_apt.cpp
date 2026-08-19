/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/group_norm_regbase_welford.h"
#include "arch35/group_norm_regbase_two_pass.h"
#include "arch35/group_norm_regbase_two_pass_generalized.h"
#include "arch35/group_norm_regbase_welford_generalized.h"

namespace {
#define TILINGKEY_WELFORD_PERF 1100
#define TILINGKEY_TWOPASS_PERF 1110
#define TILINGKEY_WELFORD_GENERALIZED 1120
#define TILINGKEY_TWOPASS_GENERALIZED 1130
} // namespace

extern "C" __global__ __aicore__ void group_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                 GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    // 正常路径依赖运行时用户workspace。
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    // 根据tiling key分派全载或分块归约模板。
    if (TILING_KEY_IS(TILINGKEY_WELFORD_PERF)) {
        GroupNorm::GroupNormWelford<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, userWorkspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_PERF)) {
        GroupNorm::GroupNormTwoPass<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, userWorkspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_WELFORD_GENERALIZED)) {
        GroupNorm::GroupNormWelfordGeneralized<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, userWorkspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_TWOPASS_GENERALIZED)) {
        GroupNorm::GroupNormTwoPassGeneralized<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, userWorkspace, &tilingData);
        op.Process();
    }
}
