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
 * \file scatter_nd_common_struct.h
 * \brief tiling base data
 */

#ifndef SCATTER_ND_COMMON_STRUCT_H
#define SCATTER_ND_COMMON_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace ScatterNdCommon {

constexpr uint16_t MAX_RANK_COUNT_NUM = 7;
constexpr uint16_t MAX_SHAPE_RANK_NUM = 8;

struct ScatterNdCommonSimtTilingData{
    uint64_t blockNum;
    uint32_t rankSize;
    uint64_t blockTilingSize;
    uint64_t tailBlockTilingSize;
    uint32_t ubTilingSize;
    uint64_t sliceSize;
    uint64_t outPutShape[MAX_SHAPE_RANK_NUM];
    uint64_t strideList[MAX_RANK_COUNT_NUM];
    int64_t varInAxis;
};

struct ScatterNdCommonSimtSortTilingData{
    uint64_t strideList[MAX_RANK_COUNT_NUM];
    int64_t indicesFactor;
    int64_t afterAxis;
    int64_t varInAxis;
    int64_t afterAxisFactor;
    int64_t indexRankSize;
    int64_t eachCoreAfterAxisCount;
    int64_t eachCoreIndexCount;
    int64_t tailCoreIndexCount;
    int64_t usedCoreNumBefore;
    int64_t updateLoopSize;
    int64_t updateTailNum;
};

struct ScatterNdCommonSimdSortTilingData{
    uint64_t strideList[MAX_RANK_COUNT_NUM];
    uint64_t outPutShape[MAX_SHAPE_RANK_NUM];
    int64_t eachCoreAfterAxisCount;
    int64_t indexRankSize;
    int64_t eachCoreIndexCount;
    int64_t tailCoreIndexCount;
    int64_t indicesFactor;
    int64_t indiceTailNum;
    int64_t indicesLoopSize;
    int64_t afterAxis;
    int64_t afterAxisFactor;
    int64_t usedCoreNumBefore;
    int64_t updateLoopSize;
    int64_t tailUpdateLoopSize;
    int64_t tailUpdateTailNum;
    int64_t updateTailNum;
    int64_t isSplitAfterAxis;
    int64_t singleCol;
};


}// namespace ScatterNdCommon
#endif