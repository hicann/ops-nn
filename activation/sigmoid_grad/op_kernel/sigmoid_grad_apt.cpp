/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_grad_apt.cpp
 * \brief
 */
#include "arch35/sigmoid_grad_f16.h"
#include "arch35/sigmoid_grad_bf16.h"
#include "arch35/sigmoid_grad_f32.h"

using namespace SigmoidGrad;

// y is float16, dy is float16, z is float16
#define SIGMOID_GRAD_F16_TILING_KEY 100000000000100
// y is bfloat16, dy is bfloat16, z is bfloat16
#define SIGMOID_GRAD_BF16_TILING_KEY 200000000000100
// y is float32, dy is float32, z is float32
#define SIGMOID_GRAD_F32_TILING_KEY 300000000000100

extern "C" __global__ __aicore__ void sigmoid_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe tPipe;
    if (TILING_KEY_IS(SIGMOID_GRAD_F16_TILING_KEY)) {
        SigmoidGradF16 op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SIGMOID_GRAD_BF16_TILING_KEY)) {
        SigmoidGradBf16 op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SIGMOID_GRAD_F32_TILING_KEY)) {
        SigmoidGradF32 op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    }
}
