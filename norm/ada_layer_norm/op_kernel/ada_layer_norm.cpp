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
 * \file ada_layer_norm.cpp
 * \brief
 */

#include "ada_layer_norm_base.h"
#include "ada_layer_norm_base_v1.h"

using namespace AdaLayerNormNS;

extern "C" __global__ __aicore__ void ada_layer_norm(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR weight, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

#define INIT_AND_PROCESS           \
    op.Init(&gmAddr, &tilingData); \
    op.Process()

    GmAddr gmAddr = {x, scale, shift, weight, bias, nullptr, out, nullptr, nullptr, nullptr};
    if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same_v<DTYPE_X, float>) {
            AdaLayerNormND<float, float, BASE_OP_CODE> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, half>) {
            AdaLayerNormND<half, half, BASE_OP_CODE> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            AdaLayerNormND<bfloat16_t, bfloat16_t, BASE_OP_CODE> op;
            INIT_AND_PROCESS;
        }
    }
}