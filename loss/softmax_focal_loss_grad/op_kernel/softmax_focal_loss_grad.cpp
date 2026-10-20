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
 * \file softmax_focal_loss_grad.cpp
 * \brief softmax_focal_loss_grad kernel entry
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/softmax_focal_loss_grad_nd_vec.h"
#include "arch35/softmax_focal_loss_grad_tiling_data.h"
#include "arch35/softmax_focal_loss_grad_tiling_key.h"

using namespace AscendC;

// weight 是可选输入, 缺省时 wrapper 不生成 DTYPE_WEIGHT 宏, 编译会报未定义标识符。
// 兜底取 DTYPE_PRED: 声明的组合里 weight 恒与 pred 同 dtype, 且缺省时该类型仅作模板占位。
#ifndef DTYPE_WEIGHT
#define DTYPE_WEIGHT DTYPE_PRED
#endif

template <uint64_t hasWeight>
__global__ __aicore__ void softmax_focal_loss_grad(GM_ADDR pred, GM_ADDR target, GM_ADDR dout, GM_ADDR weight,
                                                   GM_ADDR grad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftmaxFocalLossGradArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftmaxFocalLossGradArch35TilingData, tilingData, tiling);

    TPipe pipe;
    SoftmaxFocalLossGrad::SoftmaxFocalLossGradND<DTYPE_PRED, DTYPE_WEIGHT, hasWeight> op;
    op.Init(pred, target, dout, weight, grad, &tilingData, &pipe);
    op.Process();
}
