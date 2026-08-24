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
 * \file bn_infer.cpp
 * \brief BNInfer AscendC 950 kernel entry.
 */
#include "kernel_operator.h"
#include "arch35/bn_infer_infer.h"
#include "arch35/bn_infer_infer_last_channel.h"
#include "arch35/bn_infer_infer_last_channel_continuous_a.h"
#include "arch35/bn_infer_infer_last_channel_small_a.h"
#include "arch35/bn_infer_infer_small_ab1.h"

using namespace AscendC;
using namespace BNInferOps;

namespace {
#define TILING_KEY_LAST_CHANNEL 900000
#define TILING_KEY_LAST_CHANNEL_CONTINUOUS_A 901000
#define TILING_KEY_LAST_CHANNEL_SMALL_A 902000
#define TILING_KEY_INFER 910000
#define TILING_KEY_INFER_SMALL_AB1 911000
} // namespace

extern "C" __global__ __aicore__ void bn_infer(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean, GM_ADDR variance,
                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    REGISTER_TILING_DEFAULT(BNInferTilingData);
    if (TILING_KEY_IS(TILING_KEY_LAST_CHANNEL)) {
        GET_TILING_DATA_WITH_STRUCT(BNInferLastChannelTilingData, tilingDataIn, tiling);
        BNInferLastChannel<DTYPE_X, DTYPE_SCALE, DTYPE_MEAN> op(&tilingDataIn);
        op.Init(x, scale, offset, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LAST_CHANNEL_CONTINUOUS_A)) {
        GET_TILING_DATA_WITH_STRUCT(BNInferLastChannelTilingData, tilingDataIn, tiling);
        BNInferLastChannelContinuousA<DTYPE_X, DTYPE_SCALE, DTYPE_MEAN> op(&tilingDataIn);
        op.Init(x, scale, offset, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LAST_CHANNEL_SMALL_A)) {
        GET_TILING_DATA_WITH_STRUCT(BNInferLastChannelTilingData, tilingDataIn, tiling);
        BNInferLastChannelSmallA<DTYPE_X, DTYPE_SCALE, DTYPE_MEAN> op(&tilingDataIn);
        op.Init(x, scale, offset, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_INFER)) {
        GET_TILING_DATA_WITH_STRUCT(BNInferTilingData, tilingDataIn, tiling);
        BNInfer<DTYPE_X, DTYPE_SCALE, DTYPE_MEAN> op(&tilingDataIn);
        op.Init(x, scale, offset, mean, variance, y, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_INFER_SMALL_AB1)) {
        GET_TILING_DATA_WITH_STRUCT(BNInferTilingData, tilingDataIn, tiling);
        BNInferSmallAB1<DTYPE_X, DTYPE_SCALE, DTYPE_MEAN> op(&tilingDataIn);
        op.Init(x, scale, offset, mean, variance, y, &pipe);
        op.Process();
    }
}
