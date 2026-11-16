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
 * \file apply_came_part1.cpp
 * \brief
 */

#include "arch35/apply_came_part1_tiling_data.h"
#include "arch35/apply_came_part1_tiling_key.h"
#include "arch35/apply_came_part1_batch.h"
#include "arch35/apply_came_part1_fp32.h"
#include "arch35/apply_came_part1_fp16.h"
#include "arch35/apply_came_part1_post.h"

using namespace ApplyCamePart1;

__global__ __aicore__ void apply_came_part1(GM_ADDR grad, GM_ADDR eps, GM_ADDR sum_grad_r, GM_ADDR sum_grad_c,
                                            GM_ADDR sum_grad_rc, GM_ADDR workspace, GM_ADDR tiling)
{
    using D_T = DTYPE_GRAD;
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    ENABLE_DETERMINISTIC();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ApplyCamePart1TilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyCamePart1TilingData, tilingData, tiling);

    const int64_t sliceCoreNum = tilingData.nCoreNum * tilingData.mCoreNum;
    const int64_t batchCoreNum = tilingData.batchCoreNum > 0 ? tilingData.batchCoreNum : 1;
    const int64_t batchPerGroup = ApplyCamePart1::CeilDivApplyCamePart1(tilingData.batchCount, batchCoreNum);
    const int64_t groupIdx = static_cast<int64_t>(GetBlockIdx()) / sliceCoreNum;
    const int64_t batchStart = groupIdx * batchPerGroup;
    const int64_t localBlockIdx = static_cast<int64_t>(GetBlockIdx()) % sliceCoreNum;
    const bool isGroupPostBlock = (localBlockIdx == 0);
    const bool isRowPostBlock = (localBlockIdx % tilingData.mCoreNum == 0);
    const bool isColumnPostBlock = (localBlockIdx < tilingData.mCoreNum);
    const int64_t rowShardIdx = localBlockIdx / tilingData.mCoreNum;
    const int64_t columnShardIdx = localBlockIdx % tilingData.mCoreNum;

    if (tilingData.directBatchMode != 0) {
        ApplyCamePart1::ApplyCamePart1Batch<D_T> batchOp;
        batchOp.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, &tilingData);
        batchOp.Process(tilingData.batchCount, tilingData.usedCoreNum);
        return;
    }

    if constexpr (sizeof(D_T) == sizeof(float)) {
        ApplyCamePart1::ApplyCamePart1FP32<float> op;
        op.InitOnce(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        ApplyCamePart1::ApplyCamePart1Post<float> opPost;
        const bool directWidth2R = ApplyCamePart1UseDirectWidth2R(tilingData.N, tilingData.M, tilingData.mCoreNum,
                                                                  tilingData.tileSize);
        // The zeroing helpers reuse the shared UB arena. Load eps for the post
        // phase only after zeroing, otherwise a long-lived scalar can be lost
        // when the compiler spills it into the reused arena.
        op.ZeroWorkspace(userWS);
        op.ZeroOutputs(sum_grad_r, sum_grad_c, sum_grad_rc, tilingData.batchCount);
        SyncAll();
        opPost.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData, op.GetUbArena());
        // Keep the iteration count identical across groups for paired barriers.
        for (int64_t iter = 0; iter < batchPerGroup; ++iter) {
            const int64_t batchIdx = batchStart + iter;
            const bool active = batchIdx < tilingData.batchCount;
            if (active) {
                op.InitBatch(batchIdx);
                op.Process();
            }
            SyncAll();
            // R and C are intentionally finalized by independent shard
            // leaders. ProcessBatch below owns only the scalar RC reduction.
            if (active && isRowPostBlock && !directWidth2R) {
                opPost.ProcessRShard(batchIdx, rowShardIdx);
            }
            if (active && isColumnPostBlock) {
                opPost.ProcessCShard(batchIdx, columnShardIdx);
            }
            if (active && isGroupPostBlock) {
                opPost.ProcessBatch(batchIdx);
            }
            SyncAll();
        }
    } else {
        ApplyCamePart1::ApplyCamePart1FP16<D_T> op;
        op.InitOnce(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData);
        ApplyCamePart1::ApplyCamePart1Post<float> opPost;
        op.ZeroWorkspace(userWS);
        op.ZeroOutputs(sum_grad_r, sum_grad_c, sum_grad_rc, tilingData.batchCount);
        SyncAll();
        opPost.Init(grad, eps, sum_grad_r, sum_grad_c, sum_grad_rc, userWS, &tilingData, op.GetUbArena());
        for (int64_t iter = 0; iter < batchPerGroup; ++iter) {
            const int64_t batchIdx = batchStart + iter;
            const bool active = batchIdx < tilingData.batchCount;
            if (active) {
                op.InitBatch(batchIdx);
                op.Process();
            }
            SyncAll();
            if (active && isRowPostBlock) {
                opPost.ProcessRShard(batchIdx, rowShardIdx);
            }
            if (active && isColumnPostBlock) {
                opPost.ProcessCShard(batchIdx, columnShardIdx);
            }
            if (active && isGroupPostBlock) {
                opPost.ProcessBatch(batchIdx);
            }
            SyncAll();
        }
    }
}
