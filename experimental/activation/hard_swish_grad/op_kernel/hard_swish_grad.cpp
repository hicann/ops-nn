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
 * \file hard_swish_grad.cpp
 * \brief HardSwishGrad 算子 kernel 入口
 */

#include "hard_swish_grad.h"

enum class HardSwishGradTilingKey : uint32_t {
    TILING_KEY_HARDSWISHGRAD_FP16 = 0,
    TILING_KEY_HARDSWISHGRAD_FP32 = 1,
    TILING_KEY_HARDSWISHGRAD_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void hard_swish_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSwishGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSwishGradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(HardSwishGradTilingKey::TILING_KEY_HARDSWISHGRAD_FP16)) {
        NsHardSwishGrad::HardSwishGrad<half> op;
        op.Init(grad, x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(HardSwishGradTilingKey::TILING_KEY_HARDSWISHGRAD_FP32)) {
        NsHardSwishGrad::HardSwishGrad<float> op;
        op.Init(grad, x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(HardSwishGradTilingKey::TILING_KEY_HARDSWISHGRAD_BF16)) {
        NsHardSwishGrad::HardSwishGrad<bfloat16_t> op;
        op.Init(grad, x, y, &tilingData);
        op.Process();
    }
}
