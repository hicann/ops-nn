/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lib/matmul_intf.h"
#include "chunk_loss_kernel.h"
#include "chunk_loss_tiling_data.h"

extern "C" __global__ __aicore__ void chunk_loss(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR loss,
                                                 GM_ADDR maxOut, GM_ADDR sumOut, GM_ADDR dH, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLossTilingData, tilingData, tiling);
    // Soft FFTS base like SoftSync Launch (aclrtGetHardwareSyncAddr via tiling / fallback).
    uint64_t syncAddr = tilingData.syncBaseAddr ? tilingData.syncBaseAddr : 0xfee00000000ull;
    AscendC::SetSyncBaseAddr(syncAddr);
    chunkloss::Params params;
    params.hidden = hidden;
    params.weight = weight;
    params.labels = labels;
    params.loss = loss;
    params.maxOut = maxOut;
    params.sumOut = sumOut;
    params.dH = dH;
    params.workspace = AscendC::GetUserWorkspace(workspace);
    params.m = tilingData.m;
    params.h = tilingData.h;
    params.v = tilingData.v;
    params.aicNum = tilingData.aicNum;
    params.nSplit = 1;
    params.skipMaxSum = 0;
    params.skipCe = 0;
    // ShortV / SingleN (nTiles <= kStages=2): Pass2 rereads Pass1 logit GM — no ZFull.
    // M=1 + logitAsZ bistable dH flake (Gen case1): use rematZ instead (no ZFull needed).
    constexpr uint32_t kTileN = 256u;
    constexpr uint32_t kStages = 2u;
    uint32_t nTiles = (tilingData.v + kTileN - 1u) / kTileN;
    bool shortV = (nTiles > 0u && nTiles <= kStages);
    params.rematZ = (shortV && tilingData.m <= 1u) ? 1u : 0u;
    params.logitAsZ = (shortV && tilingData.m > 1u) ? 1u : 0u;
    params.alpha = tilingData.alpha;
    params.ignoreIndex = tilingData.ignoreIndex;
    // Per-dtype binaries via DTYPE_HIDDEN (opc); avoid instantiating half+bf16 in one TU.
#if defined(DTYPE_HIDDEN)
    chunkloss::Kernel<DTYPE_HIDDEN> op;
    op(params);
#endif
}
