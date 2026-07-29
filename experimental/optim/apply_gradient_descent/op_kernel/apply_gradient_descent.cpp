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
 * \file apply_gradient_descent.cpp
 * \brief apply_gradient_descent classic (ascend910b) kernel entry.
 *        Dispatches on the dtype tiling key: 1 = float16, 2 = float32, 3 = bfloat16.
 */

#include "kernel_operator.h"
#include "apply_gradient_descent.h"
#include "apply_gradient_descent_tiling_data.h"

using namespace AscendC;
using namespace ApplyGradientDescentClassic;

namespace {
template <typename T>
__aicore__ inline void RunApplyGradientDescent(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                               const ApplyGradientDescentTilingData& tilingData, TPipe* pipe)
{
    ApplyGradientDescentKernel<T> op(pipe);
    op.Init(var, alpha, delta, var_out, tilingData);
    op.Process();
}
} // namespace

#ifdef __CCE_UT_TEST__
extern "C" __global__ __aicore__ void apply_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(ApplyGradientDescentTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        RunApplyGradientDescent<half>(var, alpha, delta, var_out, tilingData, &pipe);
    } else if (TILING_KEY_IS(2)) {
        RunApplyGradientDescent<float>(var, alpha, delta, var_out, tilingData, &pipe);
    } else if (TILING_KEY_IS(3)) {
        RunApplyGradientDescent<bfloat16_t>(var, alpha, delta, var_out, tilingData, &pipe);
    }
}
#else
extern "C" __global__ __aicore__ void apply_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    REGISTER_TILING_DEFAULT(ApplyGradientDescentTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyGradientDescentTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        RunApplyGradientDescent<half>(var, alpha, delta, var_out, tilingData, &pipe);
    } else if (TILING_KEY_IS(2)) {
        RunApplyGradientDescent<float>(var, alpha, delta, var_out, tilingData, &pipe);
    } else if (TILING_KEY_IS(3)) {
        RunApplyGradientDescent<bfloat16_t>(var, alpha, delta, var_out, tilingData, &pipe);
    }
}
#endif
