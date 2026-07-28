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
 * \file poisson_nll_loss.cpp
 * \brief PoissonNllLoss kernel entry (hand-written, reduction=none stage-1).
 */

#include "arch35/poisson_nll_loss.h"

#ifdef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void poisson_nll_loss(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PoissonNllLossTilingData, tilingData, tiling);
    NsPoissonNllLoss::KernelPoissonNllLoss<DTYPE_INPUT_X, float, 0> op;
    op.Init(input_x, target, loss, workspace, &tilingData);
    op.Process();
}
#else
template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void poisson_nll_loss(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PoissonNllLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(PoissonNllLossTilingData, tilingData, tiling);
    NsPoissonNllLoss::KernelPoissonNllLoss<D_T_X, float, BUFFER_MODE> op;
    op.Init(input_x, target, loss, workspace, &tilingData);
    op.Process();
}
#endif
