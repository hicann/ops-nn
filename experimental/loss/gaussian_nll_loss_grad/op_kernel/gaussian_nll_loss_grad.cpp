/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gaussian_nll_loss_grad.h"

#ifndef DTYPE_GRADOUTPUT
#define DTYPE_GRADOUTPUT half
#endif

extern "C" __global__ __aicore__ void gaussian_nll_loss_grad(GM_ADDR gradOutput, GM_ADDR input, GM_ADDR target,
                                                             GM_ADDR var, GM_ADDR gradInput, GM_ADDR gradVar,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GaussianNllLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GaussianNllLossGradTilingData, tilingData, tiling);
    (void)workspace;
    AscendC::TPipe pipe;
    NsGaussianNllLossGrad::KernelGaussianNllLossGrad<DTYPE_GRADOUTPUT> op;
    op.Init(gradOutput, input, target, var, gradInput, gradVar, &tilingData, pipe);
    op.Process();
}
