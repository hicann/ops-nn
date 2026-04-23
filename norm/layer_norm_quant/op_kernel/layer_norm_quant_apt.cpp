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
 * \file layer_norm_quant_apt.cpp
 * \brief
 */

#include "./arch35/layer_norm_quant_regbase_normal.h"
#include "./arch35/layer_norm_quant_regbase_spilt_d.h"

extern "C" __global__ __aicore__ void layer_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                                                       GM_ADDR offset, GM_ADDR z, GM_ADDR scale_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(LayerNormQuantRegTilingData, tilingData, tiling);
    if (TILING_KEY_IS(2300000000)) { // half & SliceCompute
        KernelLayerNormQuantSplitD<half> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2310000000)) { // half & FastCompute
        KernelLayerNormQuantNormal<half> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2200000000)) { // bf16 & SliceCompute
        KernelLayerNormQuantSplitD<bfloat16_t> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2210000000)) { // bf16 & FastCompute
        KernelLayerNormQuantNormal<bfloat16_t> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2400000000)) { // float & SliceCompute
        KernelLayerNormQuantSplitD<float> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(2410000000)) { // float & FastCompute
        KernelLayerNormQuantNormal<float> op;
        op.Init(x, gamma, beta, scale, offset, z, tilingData);
        op.Process();
    }
}