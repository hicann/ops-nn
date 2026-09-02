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

#include "sigmoid_focal_loss_grad_kernel.h"
#include "sigmoid_focal_loss_grad_struct.h"
#include "sigmoid_focal_loss_grad_tiling_struct.h"

template <uint64_t hasWeight>
__global__ __aicore__ void sigmoid_focal_loss_grad(GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight,
                                                   GM_ADDR grad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (pred == nullptr || target == nullptr || dout == nullptr || grad == nullptr) {
        return;
    }
    if constexpr (hasWeight == 1) {
        if (weight == nullptr) {
            return;
        }
    }

    GET_TILING_DATA_WITH_STRUCT(SigmoidFocalLossGradTilingData, tilingData, tiling);
    if (tilingData.dim0 <= 0 || tilingData.blockNum <= 0 || tilingData.ubFormer <= 0) {
        return;
    }
    if constexpr (hasWeight == 1) {
        if (tilingData.weightDtype == 0) {
            SigmoidFocalLossGradKernel<DTYPE_PRED, DTYPE_DOUT, half, true> kernel;
            kernel.Init(pred, target, dout, weight, grad, &tilingData);
            kernel.Process();
        } else if (tilingData.weightDtype == 1) {
            SigmoidFocalLossGradKernel<DTYPE_PRED, DTYPE_DOUT, float, true> kernel;
            kernel.Init(pred, target, dout, weight, grad, &tilingData);
            kernel.Process();
        } else {
            return;
        }
    } else {
        SigmoidFocalLossGradKernel<DTYPE_PRED, DTYPE_DOUT, half, false> kernel;
        kernel.Init(pred, target, dout, weight, grad, &tilingData);
        kernel.Process();
    }
}
