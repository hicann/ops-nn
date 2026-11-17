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
 * \file chunk_loss_arch35.cpp
 * \brief ChunkLoss AscendC kernel 入口
 *
 * 从 TilingData 填充 Params，按 tileM / L0C→UB 场景分发 arch35 Kernel 模板。
 * 小中规模 (M*V 低于阈值) 强制 GM Pass2 + Vector float dH；超大 shape 走 Cube Pass2。
 */
#include "lib/matmul_intf.h"
#include "chunk_loss_kernel_arch35.h"
#include "../chunk_loss_tiling_data.h"

extern "C" __global__ __aicore__ void chunk_loss(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR loss,
                                                 GM_ADDR maxOut, GM_ADDR sumOut, GM_ADDR dH, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkLossTilingData, tilingData, tiling);

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
    params.nSplit = tilingData.nSplit < 1u ? 1u : tilingData.nSplit;
    params.deterministic = tilingData.deterministic;
    params.alpha = tilingData.alpha;
    params.ignoreIndex = tilingData.ignoreIndex;

    uint32_t tileM = tilingData.tileM;
    if (tileM != 32u && tileM != 64u && tileM != 112u && tileM != 128u) {
        tileM = chunkloss::SelectTileM(params.m, params.aicNum, params.v);
    }
    auto scen = chunkloss::SelectScenario(params.m, params.h, params.v, params.aicNum, params.deterministic);
    bool l0c = scen.l0cToUb && (params.nSplit <= 1u);
    {
        uint32_t rem = params.m % tileM;
        if (l0c && params.m > tileM && rem != 0u && rem < 16u) {
            l0c = false;
        }
    }
    // Vector float dH 需要 GM Pass2 与 nSplit==1；超大 MV 保留 Cube Pass2。
    constexpr uint64_t kCubeOnlyMV = 2000000000ull;
    uint64_t mv = static_cast<uint64_t>(params.m) * static_cast<uint64_t>(params.v);
    if (mv < kCubeOnlyMV) {
        l0c = false;
        params.nSplit = 1u;
    }

#if defined(DTYPE_HIDDEN)
#define CHUNKLOSS_DISPATCH_TILE(TM)                        \
    do {                                                   \
        if (l0c) {                                         \
            chunkloss::Kernel<DTYPE_HIDDEN, TM, true> op;  \
            op(params);                                    \
        } else {                                           \
            chunkloss::Kernel<DTYPE_HIDDEN, TM, false> op; \
            op(params);                                    \
        }                                                  \
    } while (0)
    if (tileM <= 32u) {
        CHUNKLOSS_DISPATCH_TILE(32);
    } else if (tileM <= 64u) {
        CHUNKLOSS_DISPATCH_TILE(64);
    } else if (tileM <= 112u) {
        CHUNKLOSS_DISPATCH_TILE(112);
    } else {
        CHUNKLOSS_DISPATCH_TILE(128);
    }
#undef CHUNKLOSS_DISPATCH_TILE
#else
#define CHUNKLOSS_DISPATCH_DTYPE_TILE(DT, TM)    \
    do {                                         \
        if (l0c) {                               \
            chunkloss::Kernel<DT, TM, true> op;  \
            op(params);                          \
        } else {                                 \
            chunkloss::Kernel<DT, TM, false> op; \
            op(params);                          \
        }                                        \
    } while (0)
    if (TILING_KEY_IS(0)) {
        if (tileM <= 32u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(half, 32);
        } else if (tileM <= 64u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(half, 64);
        } else if (tileM <= 112u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(half, 112);
        } else {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(half, 128);
        }
    } else if (TILING_KEY_IS(1)) {
        if (tileM <= 32u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(bfloat16_t, 32);
        } else if (tileM <= 64u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(bfloat16_t, 64);
        } else if (tileM <= 112u) {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(bfloat16_t, 112);
        } else {
            CHUNKLOSS_DISPATCH_DTYPE_TILE(bfloat16_t, 128);
        }
    }
#undef CHUNKLOSS_DISPATCH_DTYPE_TILE
#endif
}
