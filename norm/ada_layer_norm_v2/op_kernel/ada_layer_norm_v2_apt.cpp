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
 * \file ada_layer_norm_v2_apt.cpp
 * \brief
 */

#include "../ada_layer_norm/arch35/ada_layer_norm_impl.h"

using namespace AdaLayerNormNS;

extern "C" __global__ __aicore__ void ada_layer_norm_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR weight, GM_ADDR bias, GM_ADDR out, 
    GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

#define INIT_AND_PROCESS             \
    op.InitV2(&gmAddr, &tilingData); \
    op.Process()

    GmAddr gmAddr = {x, scale, shift, weight, bias, nullptr, out, mean, rstd, nullptr};
    if (TILING_KEY_IS(11)) {
        if constexpr (std::is_same_v<DTYPE_X, half>) {
            AdaLayerNormFullLoad<half, half, half, BASE_V2_OP_CODE> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            AdaLayerNormFullLoad<bfloat16_t, bfloat16_t, bfloat16_t, BASE_V2_OP_CODE> op;
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(12)) {
        AdaLayerNormFullLoad<DTYPE_X, float, DTYPE_X, BASE_V2_OP_CODE> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        if constexpr (std::is_same_v<DTYPE_X, half>) {
            AdaLayerNormWelford<half, half, half, BASE_V2_OP_CODE> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            AdaLayerNormWelford<bfloat16_t, bfloat16_t, bfloat16_t, BASE_V2_OP_CODE> op;
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(22)) {
        AdaLayerNormWelford<DTYPE_X, float, DTYPE_X, BASE_V2_OP_CODE> op;
        INIT_AND_PROCESS;
    }
}