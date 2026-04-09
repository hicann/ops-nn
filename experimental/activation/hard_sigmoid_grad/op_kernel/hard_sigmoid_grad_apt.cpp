/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_sigmoid_grad_apt.cpp
 * \brief HardSigmoidGrad kernel entry (arch35)
 *
 * Template parameters (from hard_sigmoid_grad_tiling_key.h):
 *   - schMode: scheduling mode (0=float32, 1=float16, 2=bfloat16)
 */

#include "arch35/hard_sigmoid_grad.h"
#include "arch35/hard_sigmoid_grad_bf16.h"

template <uint32_t schMode>
__global__ __aicore__ void hard_sigmoid_grad(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSigmoidGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSigmoidGradTilingData, tilingData, tiling);

    if constexpr (schMode == HARD_SIGMOID_GRAD_MODE_FLOAT32) {
        NsHardSigmoidGrad::HardSigmoidGrad<float> op;
        op.Init(gradOutput, self, gradInput, &tilingData);
        op.Process();
    } else if constexpr (schMode == HARD_SIGMOID_GRAD_MODE_FLOAT16) {
        NsHardSigmoidGrad::HardSigmoidGrad<half> op;
        op.Init(gradOutput, self, gradInput, &tilingData);
        op.Process();
    } else if constexpr (schMode == HARD_SIGMOID_GRAD_MODE_BFLOAT16) {
        NsHardSigmoidGrad::HardSigmoidGradBf16 op;
        op.Init(gradOutput, self, gradInput, &tilingData);
        op.Process();
    }
}
