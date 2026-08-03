/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file nll_loss_grad.cpp
 * \brief NllLossGrad 算子 kernel 入口
 */

#include "nll_loss_grad.h"

// schMode 编码 浮点dtype × target dtype 组合，顺序与 op proto 一致：
//   0: float32/int32   1: bf16/int32    2: float32/int64
//   3: bf16/int64      4: float32/uint8 5: bf16/uint8
//   6: float16/int32   7: float16/int64 8: float16/uint8

template <uint32_t schMode>
__global__ __aicore__ void nll_loss_grad(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight,
                                         GM_ADDR total_weight, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NllLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(NllLossGradTilingData, tilingData, tiling);

    if constexpr (schMode == 0) {
        NsNllLossGrad::NllLossGrad<float, int32_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsNllLossGrad::NllLossGrad<bfloat16_t, int32_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsNllLossGrad::NllLossGrad<float, int64_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 3) {
        NsNllLossGrad::NllLossGrad<bfloat16_t, int64_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 4) {
        NsNllLossGrad::NllLossGrad<float, uint8_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 5) {
        NsNllLossGrad::NllLossGrad<bfloat16_t, uint8_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 6) {
        NsNllLossGrad::NllLossGrad<half, int32_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 7) {
        NsNllLossGrad::NllLossGrad<half, int64_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    } else if constexpr (schMode == 8) {
        NsNllLossGrad::NllLossGrad<half, uint8_t> op;
        op.Init(x, y_grad, target, weight, total_weight, x_grad, &tilingData);
        op.Process();
    }
}
