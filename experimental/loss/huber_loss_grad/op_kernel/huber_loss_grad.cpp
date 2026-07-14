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
 * \file huber_loss_grad.cpp
 * \brief HuberLossGrad算子的kernel入口函数
 *
 * NPU路径：根据def的DataType编译变体注入的DTYPE_PREDICTIONS宏决定数据类型。
 * UT(CPU模拟)路径：schMode模板参数(0=FLOAT/1=FLOAT16/2=BF16)区分数据类型。
 */

#include "huber_loss_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void huber_loss_grad(GM_ADDR predictions, GM_ADDR targets, GM_ADDR grad_output, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HuberLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HuberLossGradTilingData, tilingData, tiling);

#ifdef __CCE_KT_TEST__
    // UT（CPU 模拟）路径：schMode 模板参数区分数据类型，支持一份二进制测多种 dtype
    if constexpr (schMode == 0) {
        NsHuberLossGrad::KernelHuberLossGrad<float> op;
        op.Init(predictions, targets, grad_output, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsHuberLossGrad::KernelHuberLossGrad<half> op;
        op.Init(predictions, targets, grad_output, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsHuberLossGrad::KernelHuberLossGrad<bfloat16_t> op;
        op.Init(predictions, targets, grad_output, workspace, &tilingData);
        op.Process();
    }
#else
    // NPU 路径：编译系统按 def DataType 注入 DTYPE_PREDICTIONS 宏，编译出 3 个 .o
    NsHuberLossGrad::KernelHuberLossGrad<DTYPE_PREDICTIONS> op;
    op.Init(predictions, targets, grad_output, workspace, &tilingData);
    op.Process();
#endif
}
