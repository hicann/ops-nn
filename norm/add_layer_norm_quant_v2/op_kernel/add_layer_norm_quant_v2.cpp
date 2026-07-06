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
 * \file add_layer_norm_quant_v2.cpp
 * \brief
 */

#include "../add_layer_norm_quant/add_layer_norm_static_quant_normal_kernel.h"
#include "../add_layer_norm_quant/add_layer_norm_dual_static_quant_single_row_kernel.h"
#include "../add_layer_norm_quant/add_layer_norm_sole_static_quant_single_row_kernel.h"
#include "../add_layer_norm_quant/add_layer_norm_quant_split_d.h"

extern "C" __global__ __aicore__ void add_layer_norm_quant_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                              GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2,
                                                              GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1,
                                                              GM_ADDR y2, GM_ADDR x, GM_ADDR layernormRes,
                                                              GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                                                               \
    op.Init(x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x, layernormRes, outScale1, \
            outScale2, usrWorkspace, &tilingData);                                                                     \
    op.Process()

    // NormalStatic
    if (TILING_KEY_IS(1000)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1001)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1002)) {
        KernelAddLayerNormStaticQuantNormal<half, half, 1002> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2000)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2001)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2002)) {
        KernelAddLayerNormStaticQuantNormal<float, float, 2002> op(&pipe);
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002))
    } else if (TILING_KEY_IS(3000)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3000> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3001)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3001> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3002)) {
        KernelAddLayerNormStaticQuantNormal<bfloat16_t, bfloat16_t, 3002> op(&pipe);
        INIT_AND_PROCESS;
#endif
    }

    // SingleRowStatic
    else if (TILING_KEY_IS(1020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<half, 1020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<half, 1021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(1022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<half, 1022> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<float, 2020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<float, 2021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(2022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<float, 2022> op(&pipe);
            INIT_AND_PROCESS;
        }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002))
    } else if (TILING_KEY_IS(3020)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<bfloat16_t, 3020> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3021)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<bfloat16_t, 3021> op(&pipe);
            INIT_AND_PROCESS;
        }
    } else if (TILING_KEY_IS(3022)) {
        if (tilingData.scaleOffsetMode >= 200) {
            KernelAddLayerNormDualStaticQuantSingleRowV2<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        } else {
            KernelAddLayerNormSoleStaticQuantSingleRowV2<bfloat16_t, 3022> op(&pipe);
            INIT_AND_PROCESS;
        }
#endif
    }

    // SplitD Static (SLICE_EXT = 50)
    else if (TILING_KEY_IS(1050)) {
        KernelAddLayerNormQuantSplitD<half, half, 1050> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1051)) {
        KernelAddLayerNormQuantSplitD<half, half, 1051> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1052)) {
        KernelAddLayerNormQuantSplitD<half, half, 1052> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2050)) {
        KernelAddLayerNormQuantSplitD<float, float, 2050> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2051)) {
        KernelAddLayerNormQuantSplitD<float, float, 2051> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2052)) {
        KernelAddLayerNormQuantSplitD<float, float, 2052> op(&pipe);
        INIT_AND_PROCESS;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002))
    } else if (TILING_KEY_IS(3050)) {
        KernelAddLayerNormQuantSplitD<bfloat16_t, bfloat16_t, 3050> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3051)) {
        KernelAddLayerNormQuantSplitD<bfloat16_t, bfloat16_t, 3051> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3052)) {
        KernelAddLayerNormQuantSplitD<bfloat16_t, bfloat16_t, 3052> op(&pipe);
        INIT_AND_PROCESS;
#endif
    }
}