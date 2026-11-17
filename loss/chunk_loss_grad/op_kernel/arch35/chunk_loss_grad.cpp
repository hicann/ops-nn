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

#include "chunk_loss_grad_tiling_data_arch35.h"
#include "chunk_loss_grad_fused_arch35.h"
#include "chunk_loss_grad_legacy.h"
#include "chunk_loss_grad_batch.h"
#include "chunk_loss_grad_direct.h"
#include "chunk_loss_grad_p_gm.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void chunk_loss_grad(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels,
                                                      GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR gradOutput,
                                                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkLossGradArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLossGradArch35TilingData, tilingData, tiling);
    if (TILING_KEY_IS(2)) {
        AscendC::InitSocState();
        if constexpr (ChunkLossGradOptions::BUILD_MASK & ChunkLossGradOptions::P_VIA_GM) {
            ChunkLossGradPGm::GemmUniversal<DTYPE_HIDDEN, ChunkLossGradArch35TilingData> op;
            op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
            op.Process();
        } else {
            ChunkLossGradDirect::GemmUniversal<DTYPE_HIDDEN, ChunkLossGradArch35TilingData> op;
            op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        ChunkLossGradLegacy::Kernel<DTYPE_HIDDEN, ChunkLossGradArch35TilingData> op;
        op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        if (tilingData.scheduleMode == 3) {
            ChunkLossGradBatch::Kernel<DTYPE_HIDDEN, ChunkLossGradArch35TilingData> op;
            op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
            op.Process();
        } else {
            AscendC::InitSocState();
            ChunkLossGradOpt::Kernel<DTYPE_HIDDEN, ChunkLossGradArch35TilingData> op;
            op.Init(hidden, weight, labels, softmaxMax, softmaxSum, gradOutput, output, workspace, tilingData);
            op.Process();
        }
    }
}
