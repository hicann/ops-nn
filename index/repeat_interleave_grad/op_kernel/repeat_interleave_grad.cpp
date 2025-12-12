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
 * \file repeat_interleave_grad.cpp
 * \brief
 */
#include "repeat_interleave_grad.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void repeat_interleave_grad(
    GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
    const RepeatInterleaveGradTilingData* __restrict tiling_data = &tiling_data_in;
    if (TILING_KEY_IS(0)) {
        KernelRepeatInterleaveGrad<half, int32_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelRepeatInterleaveGrad<bfloat16_t, int32_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelRepeatInterleaveGrad<float, int32_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelRepeatInterleaveGrad<half, int64_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        KernelRepeatInterleaveGrad<bfloat16_t, int64_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        KernelRepeatInterleaveGrad<float, int64_t> op;
        op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
        op.Process();
    }
}
