/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "huber_loss.h"

#ifndef DTYPE_PREDICTIONS
#define DTYPE_PREDICTIONS half
#endif

extern "C" __global__ __aicore__ void huber_loss(GM_ADDR predictions, GM_ADDR targets, GM_ADDR loss, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(HuberLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(HuberLossTilingData, tilingData, tiling);
    NsHuberLoss::KernelHuberLoss<DTYPE_PREDICTIONS> op;
    op.Init(predictions, targets, loss, &tilingData);
    op.Process();
}
