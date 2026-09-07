/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_TILING_DATA_H_
#define OPS_NORM_CENTRALIZATION_TILING_DATA_H_
#include <cstdint>

struct CentralizationTilingData {
    int64_t reduceRank = 0;
    int64_t keepRank = 0;
    int64_t groupCount = 0;
    int64_t reduceCount = 0;
    int64_t totalCount = 0;
    int64_t blockFactor = 1;
    int64_t rowsPerLoop = 1;
    int64_t rowsPerBatch = 1;
    int64_t ubFactor = 0;
    int64_t bufNum = 2;
    int64_t largePath = 0;
    int64_t rowParallel = 0;
    int64_t coresPerRow = 1;
    int64_t alignedCoresPerRow = 1;
    int64_t alignedReduce = 0;
    int64_t contiguousGeneric = 0;
    int64_t contiguousOuterCount = 1;
    int64_t contiguousInnerCount = 1;
    int64_t contiguousInnerTileElements = 0;
    int64_t contiguousInnerTileCount = 0;
    int64_t smallContiguous = 0;
    int64_t smallContiguousMode = 0;
    int64_t smallContiguousBlockElements = 0;
    int64_t smallContiguousMeanElements = 0;
    int64_t largeContiguousSmallInner = 0;
    int64_t largeContiguousReduceTile = 0;
    int64_t irregularVectorizable = 0;
    int64_t irregularKeepOuterRank = 0;
    int64_t irregularVectorInnerCount = 1;
    int64_t irregularVectorTileElements = 0;
    int64_t irregularVectorTileCount = 0;
    int64_t irregularKeepTileElements = 1;
    int64_t irregularKeepInnerCount = 1;
    int64_t irregularKeepTileCountPerPrefix = 1;
    int64_t irregularKeepTileTaskCount = 1;
    int64_t irregularVectorTaskCount = 1;
    int64_t useReduceOffsetTable = 0;
    int64_t reduceSegmentCount = 0;
    int64_t irregularReduceSuffixElements = 1;
    int64_t irregularReduceOuterRank = 0;
    int64_t irregularReduceOuterCount = 1;
    uint32_t genericInputBufferBytes = 0;
    uint32_t genericOutputBufferBytes = 0;
    uint32_t genericSumBufferBytes = 0;
    uint32_t largeTrailingInputBufferBytes = 0;
    uint32_t largeTrailingOutputBufferBytes = 0;
    uint32_t largeTrailingCalcBufferBytes = 0;
    uint32_t largeTrailingMeanBufferBytes = 0;
    int64_t irregularKeepOuterIndexStrides[8] = {};
    int64_t irregularReduceOuterIndexStrides[8] = {};
    int64_t reduceSegmentOffsets[128] = {};
    int64_t dims[8] = {};
    int64_t keepDims[8] = {};
    int64_t keepIndexStrides[8] = {};
    int64_t keepStrides[8] = {};
    int64_t reduceDims[8] = {};
    int64_t reduceIndexStrides[8] = {};
    int64_t reduceStrides[8] = {};
    int64_t reduceMask[8] = {};
};
#endif
