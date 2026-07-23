/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file selu_grad.cpp
 * \brief SeluGrad 算子 kernel 入口
 */

#include "selu_grad.h"

enum class SeluGradTilingKey : uint32_t {
    FP16 = SELUGRAD_TPL_SCH_MODE_FP16,
    FP32 = SELUGRAD_TPL_SCH_MODE_FP32,
    BF16 = SELUGRAD_TPL_SCH_MODE_BF16,
    INT32 = SELUGRAD_TPL_SCH_MODE_INT32,
    INT8 = SELUGRAD_TPL_SCH_MODE_INT8,
    UINT8 = SELUGRAD_TPL_SCH_MODE_UINT8,
};

template <uint32_t schMode>
__global__ __aicore__ void selu_grad(GM_ADDR gradients, GM_ADDR outputs, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(SeluGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SeluGradTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::FP16)) {
        NsSeluGrad::SeluGrad<half, half, false> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::FP32)) {
        NsSeluGrad::SeluGrad<float, float, false> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::BF16)) {
        NsSeluGrad::SeluGrad<bfloat16_t, float, true> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::INT32)) {
        NsSeluGrad::SeluGrad<int32_t, float, true> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::INT8)) {
        NsSeluGrad::SeluGrad<int8_t, half, true> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::UINT8)) {
        NsSeluGrad::SeluGrad<uint8_t, half, true> op;
        op.Init(pipe, gradients, outputs, y, &tilingData);
        op.Process();
    }
}
