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
 * \file cross_entropy_grad.cpp
 * \brief CrossEntropyGrad算子的kernel入口函数
 */

#include "cross_entropy_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void cross_entropy_grad(GM_ADDR logits, GM_ADDR target,
                                               GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CrossEntropyGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(CrossEntropyGradTilingData, tilingData, tiling);

#ifdef __CCE_KT_TEST__
    // UT（CPU 模拟）路径：schMode 模板参数区分数据类型，支持一份二进制测多种 dtype
    if constexpr (schMode == 0) {
        KernelCrossEntropyGrad<float> op;
        op.Init(logits, target, grad_input, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 1) {
        KernelCrossEntropyGrad<half> op;
        op.Init(logits, target, grad_input, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 2) {
        KernelCrossEntropyGrad<bfloat16_t> op;
        op.Init(logits, target, grad_input, workspace, &tilingData);
        op.Process();
    }
#else
    // NPU 路径：编译系统按 def DataType 注入 DTYPE_LOGITS 宏，编译出 3 个 .o
    KernelCrossEntropyGrad<DTYPE_LOGITS> op;
    op.Init(logits, target, grad_input, workspace, &tilingData);
    op.Process();
#endif
}
