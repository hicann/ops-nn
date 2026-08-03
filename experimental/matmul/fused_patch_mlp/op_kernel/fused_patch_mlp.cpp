/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_patch_mlp.h"

using namespace AscendC;
using namespace FusedPatchMlp;

#define FUSED_PATCH_MLP_IMPL(T, USE_MDL, PIPELINE_GELU)         \
    do {                                                        \
        KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU> op;      \
        op.Init(x, weights, biases, y, workspace, &tilingData); \
        op.Process();                                           \
    } while (0)

extern "C" __global__ __aicore__ void fused_patch_mlp(GM_ADDR x, GM_ADDR weights, GM_ADDR biases, GM_ADDR y,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    // Multilayer paths execute Matmul on AIC and GELU on the two paired AIV tasks.  The runtime task type must match
    // the SyncAll protocol in KernelFusedPatchMlp; otherwise the first launch either fails or waits for absent tasks.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(21, KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (TILING_KEY_IS(1)) {
        FUSED_PATCH_MLP_IMPL(DTYPE_X, false, false);
    } else if constexpr (TILING_KEY_IS(11)) {
        FUSED_PATCH_MLP_IMPL(DTYPE_X, true, false);
    } else if constexpr (TILING_KEY_IS(21)) {
        FUSED_PATCH_MLP_IMPL(DTYPE_X, false, false);
    } else if constexpr (TILING_KEY_IS(31)) {
        FUSED_PATCH_MLP_IMPL(DTYPE_X, true, true);
    }
}

#undef FUSED_PATCH_MLP_IMPL
