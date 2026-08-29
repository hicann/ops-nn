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
 * \file unsorted_segment_sum_tiling_data.h
 * \brief unsorted_segment_sum tiling data structs
 */

#ifndef UNSORTED_SEGMENT_SUM_TILING_DATA_H
#define UNSORTED_SEGMENT_SUM_TILING_DATA_H

#include <cstdint>

namespace UnsortedSegmentSum {

struct UnsortedSegmentSumSimtTilingData {
    uint64_t inputOuterDim;  // totalSampleNum_
    uint64_t outputOuterDim; // segmentNum_
    uint64_t innerDim;
    uint64_t maxThread;
};

struct UnsortedSegmentSumSortSimtTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint64_t maxIndexNum;
    uint64_t oneCoreUbLoopTimes;
    uint64_t tailCoreUbLoopTimes;
    uint64_t maxThread;
    uint64_t usedCoreNum;
    uint64_t sortTmpSize;
    uint64_t tailIndexNum;
    uint64_t indicesCastMode;
};

struct UnsortedSegmentSumSimdDynSortTilingData {
    uint64_t outputOuterDim; // segmentNum_
    uint64_t innerDim;
    uint64_t sTileNum;
    uint64_t aTileNum;
    uint64_t normBlockS;
    uint64_t tailBlockS;
    uint64_t normBlockA;
    uint64_t tailBlockA;
    uint64_t baseS;
    uint64_t baseA;
    uint64_t sortBaseS;
    uint64_t sortBaseA;
    uint64_t sortSharedBufSize;
    uint64_t indicesCastMode;
};

struct UnsortedSegmentSumSimdNonSortTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim; // segmentNum_
    uint64_t innerDim;
    uint64_t sTileNum;
    uint64_t aTileNum;
    uint64_t normBlockS;
    uint64_t tailBlockS;
    uint64_t normBlockA;
    uint64_t tailBlockA;
    uint64_t baseS;
    uint64_t baseA;
    uint64_t usedCoreNum;
};

struct UnsortedSegmentSumSimdSplitColTilingData {
    uint64_t inputOuterDim;  // totalSampleNum_
    uint64_t outputOuterDim; // segmentNum_
    uint64_t innerDim;
    uint64_t normBlockData;
    uint64_t tailBlockData;
    uint64_t baseS;
    uint64_t baseA;
};

struct UnsortedSegmentSumOutFlTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint64_t maxIndexNum;
    uint64_t oneCoreUbLoopTimes;
    uint64_t rowNumUb;
};

struct UnsortedSegmentSumDetermTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint32_t tmpBufferSize;
    uint32_t rowsNumInUB;
    uint32_t normalCoreProcessNum;
    uint32_t tailCoreProcessNum;
    uint32_t usedCoreNum;
};

struct UnsortedSegmentSumDeterministicBigInnerDimTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint64_t normalCoreProcessCols;
    uint64_t tailCoreProcessCols;
    uint64_t baseS;
    uint64_t baseA;
    uint32_t sortSharedBufSize;
};

struct UnsortedSegmentSumDetermSmallInnerDimTilingData {
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint32_t rowsNumInUB;
    uint32_t sortSharedBufSize;
    uint32_t usedCoreNum;
};

} // namespace UnsortedSegmentSum
#endif // UNSORTED_SEGMENT_SUM_TILING_DATA_H
