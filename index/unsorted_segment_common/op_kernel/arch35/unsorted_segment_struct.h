/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_STRUCT_H
#define UNSORTED_SEGMENT_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace UnsortedSegment {

struct UnsortedSegmentSimtTilingData{
    uint64_t inputOuterDim;  // totalSampleNum_
    uint64_t outputOuterDim; // segmentNum_
    uint64_t innerDim;
    uint64_t maxThread;
};

struct UnsortedSegmentSortSimtTilingData{
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
};

struct UnsortedSegmentSimdSplitColTilingData{
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint64_t normBlockData;
    uint64_t tailBlockData;
    uint64_t baseS;
    uint64_t baseA;
};

struct UnsortedSegmentSimdNonSortTilingData{
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
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

struct UnsortedSegmentSimdDynSortTilingData{
    uint64_t outputOuterDim;
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
    int64_t sortSharedBufSize;
};

struct UnsortedSegmentOutFlTilingData{
    uint64_t inputOuterDim;
    uint64_t outputOuterDim;
    uint64_t innerDim;
    uint64_t maxIndexNum;
    uint64_t oneCoreUbLoopTimes;
    uint64_t rowNumUb;
};

}// namespace UnsortedSegment
#endif