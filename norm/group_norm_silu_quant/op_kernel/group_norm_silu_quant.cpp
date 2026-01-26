/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "group_norm_silu_quant_b16.h"

using namespace GroupNormSiluQuant;

extern "C" __global__ __aicore__ void group_norm_silu_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR yOut, GM_ADDR meanOut, GM_ADDR rstdOut, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1031)) {
        GroupNormSiluQuant::GroupNormSiluQuantB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, quantScale, yOut, meanOut, rstdOut, userWS, &tilingData);
        op.Process();
    }
}