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
#undef APPLY_CAME_TILING_ACCESSOR
};
#endif // OPS_NN_APPLY_CAME_PART1_TILING_DATA_H
