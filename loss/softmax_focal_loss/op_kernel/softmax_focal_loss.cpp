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
 * \file softmax_focal_loss.cpp
 * \brief softmax_focal_loss kernel entry
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/softmax_focal_loss_nd_vec.h"
#include "arch35/softmax_focal_loss_tiling_data.h"
#include "arch35/softmax_focal_loss_tiling_key.h"

using namespace AscendC;

// weight 的 dtype 由 weightIsHalf 模板参数决定, 不用 DTYPE_WEIGHT ——
// 见 softmax_focal_loss_tiling_key.h 中关于 simplifiedKey 塌键的说明。
template <uint64_t hasWeight, uint64_t weightIsHalf>
__global__ __aicore__ void softmax_focal_loss(GM_ADDR pred, GM_ADDR target, GM_ADDR weight, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftmaxFocalLossArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftmaxFocalLossArch35TilingData, tilingData, tiling);

    TPipe pipe;
    if constexpr (weightIsHalf == 1) {
        SoftmaxFocalLoss::SoftmaxFocalLossND<DTYPE_PRED, half, hasWeight> op;
        op.Init(pred, target, weight, y, &tilingData, &pipe);
        op.Process();
    } else {
        SoftmaxFocalLoss::SoftmaxFocalLossND<DTYPE_PRED, float, hasWeight> op;
        op.Init(pred, target, weight, y, &tilingData, &pipe);
        op.Process();
    }
}
