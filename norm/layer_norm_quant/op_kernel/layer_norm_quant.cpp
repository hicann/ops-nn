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
 * \file layer_norm_quant.cpp
 * \brief
 */

#include "layer_norm_quant.h"

extern "C" __global__ __aicore__ void layer_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                                                       GM_ADDR offset, GM_ADDR z, GM_ADDR scale_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2000000000)) { // half & SliceCompute
        KernelLayerNormQuant<half, false> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2010000000)) { // half & FastCompute
        KernelLayerNormQuant<half, true> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16 & SliceCompute
        KernelLayerNormQuant<bfloat16_t, false> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16 & FastCompute
        KernelLayerNormQuant<bfloat16_t, true> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
#endif
}