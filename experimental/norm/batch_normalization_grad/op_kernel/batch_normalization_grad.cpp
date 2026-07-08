/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_normalization_grad.cpp
 * \brief BatchNormalizationGrad算子的kernel入口函数
 */

#include "batch_normalization_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void batch_normalization_grad(
    GM_ADDR grad_output,
    GM_ADDR input,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR save_mean,
    GM_ADDR save_invstd,
    GM_ADDR grad_input,
    GM_ADDR grad_weight,
    GM_ADDR grad_bias,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchNormalizationGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchNormalizationGradTilingData, tilingData, tiling);

#ifdef __CCE_KT_TEST__
    // UT（CPU 模拟）路径：schMode 模板参数区分数据类型，支持一份二进制测多种 dtype
    if constexpr (schMode == 0) {
        NsBatchNormalizationGrad::KernelBatchNormalizationGrad<float> op;
        op.Init(grad_output, input, weight, bias, save_mean, save_invstd,
                grad_input, grad_weight, grad_bias, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsBatchNormalizationGrad::KernelBatchNormalizationGrad<half> op;
        op.Init(grad_output, input, weight, bias, save_mean, save_invstd,
                grad_input, grad_weight, grad_bias, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsBatchNormalizationGrad::KernelBatchNormalizationGrad<bfloat16_t> op;
        op.Init(grad_output, input, weight, bias, save_mean, save_invstd,
                grad_input, grad_weight, grad_bias, workspace, &tilingData);
        op.Process();
    }
#else
    // NPU 路径：编译系统按 def DataType 注入 DTYPE_GRAD_OUTPUT 宏，编译出 3 个 .o
    NsBatchNormalizationGrad::KernelBatchNormalizationGrad<DTYPE_GRAD_OUTPUT> op;
    op.Init(grad_output, input, weight, bias, save_mean, save_invstd,
            grad_input, grad_weight, grad_bias, workspace, &tilingData);
    op.Process();
#endif
}
