/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gaussian_nll_loss.h"
#include "gaussian_nll_loss_tiling_key.h"

#ifndef DTYPE_INPUT
#define DTYPE_INPUT half
#endif

template <uint32_t reductionMode>
__global__ __aicore__ void gaussian_nll_loss(GM_ADDR input, GM_ADDR target, GM_ADDR var, GM_ADDR loss,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GaussianNllLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(GaussianNllLossTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = workspace;
    if constexpr (reductionMode != GAUSSIAN_NLL_LOSS_REDUCTION_NONE) {
        if (tilingData.blockNum > 1) {
            if (workspace == nullptr) {
                AscendC::Trap();
                return;
            }
            AscendC::SetSysWorkspace(workspace);
            userWorkspace = AscendC::GetUserWorkspace(workspace);
            if (userWorkspace == nullptr) {
                AscendC::Trap();
                return;
            }
        }
    }
    AscendC::TPipe pipe;
    NsGaussianNllLoss::KernelGaussianNllLoss<DTYPE_INPUT, reductionMode> op;
    op.Init(input, target, var, loss, userWorkspace, &tilingData, pipe);
    op.Process();
}
