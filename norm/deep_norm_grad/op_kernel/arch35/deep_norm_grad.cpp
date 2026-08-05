/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deep_norm_grad.h"

extern "C" __global__ __aicore__ void deep_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean,
                                                     GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta,
                                                     GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(DeepNormGradTilingDataArch35);
    GET_TILING_DATA_WITH_STRUCT(DeepNormGradTilingDataArch35, tilingDataIn, tiling);
    const DeepNormGradTilingDataArch35* __restrict tilingData = &tilingDataIn;
    if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        GM_ADDR userWorkspace = tilingData->gammaBetaRowSplit != 0 ? AscendC::GetUserWorkspace(workspace) : nullptr;
        DeepNormGradArch35::DeepNormGrad<DTYPE_DY> op;
        op.Init(dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, userWorkspace, tilingData);
        op.Process();
    }
}
