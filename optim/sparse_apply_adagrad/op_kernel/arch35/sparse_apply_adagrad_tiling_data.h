/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_APPLY_ADAGRAD_TILING_DATA_H_
#define SPARSE_APPLY_ADAGRAD_TILING_DATA_H_

#include <cstdint>

struct SparseApplyAdagradOptimizeTilingData {
    int64_t totalCoreNum = 0;
    int64_t blockNum = 0;
    int64_t a1_ = 0;
    int64_t r_ = 0;
    int64_t gradDim0 = 0;
    int64_t loopNumTailBlock = 0;
    int64_t loopNumFormerBlock = 0;
    int64_t formerCoreNum = 0;
    int64_t ubFormer = 0;
    int64_t ubLoop = 0;
    int64_t ubTailFormer = 0;
    bool useLocking = false;
    bool updateSlots = true;
};

struct SparseApplyAdagradOptimizeSplitRTilingData {
    int64_t totalCoreNum = 0;
    int64_t blockNum = 0;
    int64_t a1_ = 0;
    int64_t r_ = 0;
    int64_t gradDim0 = 0;
    int64_t splitblockNum = 0;
    int64_t tailBlockDataNum = 0;
    int64_t formerBlockDataNum = 0;
    int64_t ubFormer = 0;
    int64_t formerBlockUbLoop = 0;
    int64_t tailBlockUbLoop = 0;
    int64_t formerBlocktailNum = 0;
    int64_t tailBlocktailNum = 0;
    bool useLocking = false;
    bool updateSlots = true;
};

struct SparseApplyAdagradSmallDimTilingData {
    int64_t totalCoreNum = 0;
    int64_t a1_ = 0;
    int64_t r_ = 0;
    int64_t gradDim0 = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    bool useLocking = false;
    bool updateSlots = true;
};

struct SparseApplyAdagradBigDimTilingData {
    int64_t totalCoreNum = 0;
    int64_t a1_ = 0;
    int64_t r_ = 0;
    int64_t gradDim0 = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    int64_t indicesLoopNum = 0;
    int64_t indicesTailBlock = 0;
    bool useLocking = false;
    bool updateSlots = true;
};

#endif // SPARSE_APPLY_ADAGRAD_TILING_DATA_H_
