/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 
#include "smooth_l1_loss_grad_v2.h"

__global__ __aicore__ void smooth_l1_loss_grad_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR dout,
                                                   GM_ADDR gradient, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossGradV2TilingData, tilingData, tiling);

    using namespace MySmoothL1LossGradV2;
    KernelSmoothL1LossGradV2<DTYPE_PREDICT, DTYPE_LABEL, DTYPE_DOUT, DTYPE_GRADIENT> op;
    op.Init(predict, label, dout, gradient, &tilingData);
    op.Process();  
}