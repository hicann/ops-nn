/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/sigmoid_cross_entropy_with_logits_grad.h"
#include "arch35/sigmoid_cross_entropy_with_logits_grad_tiling_data.h"

using namespace AscendC;
using namespace SigmoidCrossEntropyWithLogitsGradOp;

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad(GM_ADDR predict, GM_ADDR target,
                                                                             GM_ADDR dout, GM_ADDR gradient,
                                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidCrossEntropyWithLogitsGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidCrossEntropyWithLogitsGradTilingData, tilingData, tiling);

    SigmoidBceGradKernel<DTYPE_PREDICT> op;
    op.Init(predict, target, dout, gradient, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
