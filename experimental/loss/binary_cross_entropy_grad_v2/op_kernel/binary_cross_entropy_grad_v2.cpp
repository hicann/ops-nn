/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file binary_cross_entropy_grad_v2.cpp
 * \brief
 */

#include "binary_cross_entropy_grad_v2.h"
#include "binary_cross_entropy_grad_v2_ms.h"

template <uint32_t schMode>
__global__ __aicore__ void binary_cross_entropy_grad_v2(GM_ADDR grad, GM_ADDR logits, GM_ADDR labels, GM_ADDR weight, GM_ADDR outgrad, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BinaryCrossEntropyGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(BinaryCrossEntropyGradV2TilingData, tilingData, tiling);
    // 场景1
     if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_0) {
        NsBinaryCrossEntropyGradV2::BinaryCrossEntropyGradV2<DTYPE_GRAD> op; // 算子kernel实例获取
        op.Init(grad, logits, labels, weight, outgrad, &tilingData);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }
    // 场景2
    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_1) {
        NsBinaryCrossEntropyGradV2::BinaryCrossEntropyGradV2MS<DTYPE_GRAD> op; // 算子kernel实例获取
        op.Init(grad, logits, labels, weight, outgrad, &tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
}  

