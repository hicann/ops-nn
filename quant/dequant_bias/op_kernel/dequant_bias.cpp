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
 * \file dequant_bias.cpp
 * \brief dequant_bias kernal
 */
#include "kernel_operator.h"
#include "dequant_bias_impl.h"
#include "dequant_bias_multi.h"

using namespace AscendC;
using namespace DequantBias;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

extern "C" __global__ __aicore__ void dequant_bias(
                                                   GM_ADDR x,
                                                   GM_ADDR weight_scale,
                                                   GM_ADDR activate_scale,
                                                   GM_ADDR bias,
                                                   GM_ADDR y,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10100)) {
        if (tilingData.N <= 8192) {
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, DTYPE_BIAS, DTYPE_Y, false> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, DTYPE_BIAS, DTYPE_Y, false> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10110)) {
        if (tilingData.N <= 8192) {
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, int32_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, int32_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10111)) {
        if (tilingData.N <= 8192) {
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, float, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, float, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10112)) {
        if (tilingData.N <= 8192) {
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, half, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, half, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(10113)) {
        if (tilingData.N <= 8192) {
            DequantBias::DequantBiasImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, bfloat16_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        } else { DequantBias::DequantBiasMultiImpl<DTYPE_X, DTYPE_WEIGHT_SCALE, bfloat16_t, DTYPE_Y, true> op;
            op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
            op.Process();
        }
    }
}