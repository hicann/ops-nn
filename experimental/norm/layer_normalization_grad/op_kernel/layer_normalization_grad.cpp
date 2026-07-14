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
 * \file layer_normalization_grad.cpp
 * \brief LayerNormalizationGrad 算子的 kernel 入口函数
 *
 * NPU 路径：根据 def 的 DataType 编译变体注入的 DTYPE_DY 宏决定数据类型。
 * UT（CPU 模拟）路径：schMode 模板参数（0=FLOAT/1=FLOAT16/2=BF16）区分数据类型。
 */

#include "layer_normalization_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void layer_normalization_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                                    GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LayerNormalizationGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(LayerNormalizationGradTilingData, tilingData, tiling);

#ifdef __CCE_KT_TEST__
    // UT（CPU 模拟）路径：schMode 模板参数区分数据类型，支持一份二进制测多种 dtype
    if constexpr (schMode == 0) {
        NsLayerNormalizationGrad::KernelLayerNormalizationGrad<float> op;
        op.Init(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsLayerNormalizationGrad::KernelLayerNormalizationGrad<half> op;
        op.Init(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsLayerNormalizationGrad::KernelLayerNormalizationGrad<bfloat16_t> op;
        op.Init(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, workspace, &tilingData);
        op.Process();
    }
#else
    // NPU 路径：编译系统按 def DataType 注入 DTYPE_DY 宏，编译出 3 个 .o
    NsLayerNormalizationGrad::KernelLayerNormalizationGrad<DTYPE_DY> op;
    op.Init(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, workspace, &tilingData);
    op.Process();
#endif
}
