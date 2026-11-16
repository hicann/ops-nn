/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_APPLY_CAME_PART1_TILING_DATA_H
#define OPS_NN_APPLY_CAME_PART1_TILING_DATA_H

#include <cstdint>

#ifdef __CCE_AICORE__
#define APPLY_CAME_PART1_HOST_AICORE __aicore__ inline
#else
#define APPLY_CAME_PART1_HOST_AICORE inline
#endif

APPLY_CAME_PART1_HOST_AICORE constexpr int64_t ApplyCamePart1Width2MaxChunkRows(int64_t tileSize)
{
    return tileSize * tileSize / 2;
}

APPLY_CAME_PART1_HOST_AICORE constexpr bool ApplyCamePart1UseDirectWidth2R(int64_t rowCount, int64_t columnCount,
                                                                           int64_t columnCoreCount, int64_t tileSize)
{
    return columnCount == 2 && columnCoreCount == 1 && rowCount >= ApplyCamePart1Width2MaxChunkRows(tileSize);
}

#undef APPLY_CAME_PART1_HOST_AICORE

struct ApplyCamePart1TilingData {
    int64_t N = 0;
    int64_t M = 0;
    int64_t batchCount = 1;
    int64_t nNormalCoreNum = 0;
    int64_t nTailCoreNum = 0;
    int64_t mNormalCoreNum = 0;
    int64_t mTailCoreNum = 0;
    int64_t nLoopNormCore = 0;
    int64_t nLoopTailCore = 0;
    int64_t mLoopNumCore = 0;
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t nCoreNum = 0;
    int64_t mCoreNum = 0;
    int64_t batchCoreNum = 1;
    int64_t directBatchMode = 0;
    int64_t tileSize = 64;
    int64_t ubArenaBytes = 131072;
    int64_t postUbElements = 32768;
    int64_t directUbElements = 8192;
    int64_t gradQueueBytes = 0;
    int64_t epsQueueBytes = 0;
    int64_t sumGradRQueueBytes = 0;
    int64_t sumGradCQueueBytes = 0;
    int64_t sumGradRCQueueBytes = 0;
    int64_t syncWorkQueueBytes = 0;
    int64_t directGradQueueBytes = 0;
    int64_t directEpsQueueBytes = 0;
    int64_t directSumGradRQueueBytes = 0;
    int64_t directSumGradCQueueBytes = 0;
    int64_t directCastOutputBytes = 0;
    int64_t directValueBytes = 0;
    int64_t directReduceBytes = 0;
#define APPLY_CAME_TILING_ACCESSOR(name)        \
    int64_t get_##name() const { return name; } \
    void set_##name(int64_t value) { name = value; }
    APPLY_CAME_TILING_ACCESSOR(N)
    APPLY_CAME_TILING_ACCESSOR(M)
    APPLY_CAME_TILING_ACCESSOR(batchCount)
    APPLY_CAME_TILING_ACCESSOR(nNormalCoreNum)
    APPLY_CAME_TILING_ACCESSOR(nTailCoreNum)
    APPLY_CAME_TILING_ACCESSOR(mNormalCoreNum)
    APPLY_CAME_TILING_ACCESSOR(mTailCoreNum)
    APPLY_CAME_TILING_ACCESSOR(nLoopNormCore)
    APPLY_CAME_TILING_ACCESSOR(nLoopTailCore)
    APPLY_CAME_TILING_ACCESSOR(mLoopNumCore)
    APPLY_CAME_TILING_ACCESSOR(totalCoreNum)
    APPLY_CAME_TILING_ACCESSOR(usedCoreNum)
    APPLY_CAME_TILING_ACCESSOR(nCoreNum)
    APPLY_CAME_TILING_ACCESSOR(mCoreNum)
    APPLY_CAME_TILING_ACCESSOR(batchCoreNum)
    APPLY_CAME_TILING_ACCESSOR(directBatchMode)
    APPLY_CAME_TILING_ACCESSOR(tileSize)
    APPLY_CAME_TILING_ACCESSOR(ubArenaBytes)
    APPLY_CAME_TILING_ACCESSOR(postUbElements)
    APPLY_CAME_TILING_ACCESSOR(directUbElements)
    APPLY_CAME_TILING_ACCESSOR(gradQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(epsQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(sumGradRQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(sumGradCQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(sumGradRCQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(syncWorkQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(directGradQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(directEpsQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(directSumGradRQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(directSumGradCQueueBytes)
    APPLY_CAME_TILING_ACCESSOR(directCastOutputBytes)
    APPLY_CAME_TILING_ACCESSOR(directValueBytes)
    APPLY_CAME_TILING_ACCESSOR(directReduceBytes)
#undef APPLY_CAME_TILING_ACCESSOR
};
#endif // OPS_NN_APPLY_CAME_PART1_TILING_DATA_H
