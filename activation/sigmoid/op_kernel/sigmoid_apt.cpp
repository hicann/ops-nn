/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid.cpp
 * \brief
 */
#include "arch35/sigmoid_f16.h"
#include "arch35/sigmoid_bf16.h"
#include "arch35/sigmoid_f32.h"

using namespace Sigmoid;

// x is float16, y is float16
#define SIGMOID_F16_TILING_KEY 100000000000100
// x is bfloat16, y is bfloat16
#define SIGMOID_BF16_TILING_KEY 200000000000100
// x is float32, y is float32
#define SIGMOID_F32_TILING_KEY 300000000000100

extern "C" __global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe tPipe;
    if (TILING_KEY_IS(SIGMOID_F16_TILING_KEY)) {
        SigmoidF16 op;
        op.Init(x, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SIGMOID_BF16_TILING_KEY)) {
        SigmoidBf16 op;
        op.Init(x, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SIGMOID_F32_TILING_KEY)) {
        SigmoidF32 op;
        op.Init(x, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    }
}
