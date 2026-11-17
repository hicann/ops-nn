/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "arch22/chunk_loss_grad_tiling_data_arch22.h"
#include "common/chunk_loss_grad_fused.h"
#include "arch22/chunk_loss_grad_optimized.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void chunk_loss_grad(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels,
                                                      GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR gradOutput,
                                                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkLossGradArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLossGradArch22TilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        ChunkLossGradArch22Opt::Kernel<DTYPE_HIDDEN, ChunkLossGradArch22TilingData> op;
        op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        ChunkLossGradNative::Kernel<DTYPE_HIDDEN, ChunkLossGradArch22TilingData> op;
        op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
        op.Process();
    }
}
