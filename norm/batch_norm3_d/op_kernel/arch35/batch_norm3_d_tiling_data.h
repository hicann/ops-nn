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
 * \file batch_norm3_d_tiling_data.h
 * \brief batch_norm3_d plain tiling-data structs, shared by host tiling and kernel.
 */

#ifndef NORM_BATCH_NORM3_D_TILING_DATA_H
#define NORM_BATCH_NORM3_D_TILING_DATA_H

#include <cstdint>

struct BatchNorm3DFullReduceRegbaseTilingData {
    int64_t r1 = 0;
    int64_t r0 = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t r1r0LoopCount = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLastNum = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DRAFullReduceTilingData {
    int64_t r1 = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DRARBlockSplitRTilingData {
    int64_t patternR1 = 0;
    int64_t patternA = 0;
    int64_t patternR0 = 0;
    int64_t patternAAlign = 0;
    int64_t blockSplitAxis = 0;
    int64_t formerBlockOuter = 0;
    int64_t tailBlockOuter = 0;
    int64_t blockInner = 0;
    int64_t ubFactor = 0;
    int64_t formerCoreUbSplitAxis = 0;
    int64_t formerCoreUbOuter = 0;
    int64_t formerCoreUbInner = 0;
    int64_t tailCoreUbSplitAxis = 0;
    int64_t tailCoreUbOuter = 0;
    int64_t tailCoreUbInner = 0;
    int64_t formerCoreBinaryAddQuotient = 0;
    int64_t tailCoreBinaryAddQuotient = 0;
    int64_t lastBinaryAddQuotient = 0;
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    float momentumReverse = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DWelfordRegbaseTilingData {
    int64_t r1 = 0;
    int64_t r0 = 0;
    int64_t a0 = 0;
    int64_t loopR1outer = 0;
    int64_t r1Factor = 0;
    int64_t loopR0outer = 0;
    int64_t r0Factor = 0;
    int64_t realCoreNum = 0;
    int64_t numLastCore = 0;
    int64_t aBlockFactor = 0;
    int64_t aGatherLimit = 0;
    int64_t parallelN = 0;
    int64_t processSize = 0;
    int64_t ubSize = 0;
    int64_t elemNum = 0;
    int64_t vlLenFp32 = 0;
    int64_t cutR1OrR0 = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLastNum = 0;
    int64_t binaryAddQuotient = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DRAWelfordTilingData {
    int64_t r = 0;
    int64_t rFactor = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DBlockSplitRTilingData {
    int64_t patternR = 0;
    int64_t patternA = 0;
    int64_t patternAAlign = 0;
    int64_t rUbFactor = 0;
    int64_t tBufUbFactor = 0;
    int64_t aUbFactor = 0;
    int64_t aUbLoop = 0;
    int64_t aUbTail = 0;
    int64_t formerCoreBlockFactor = 0;
    int64_t tailCoreBlockFactor = 0;
    int64_t formerCoreNums = 0;
    int64_t tailCoreNums = 0;
    int64_t tailR = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t lastBinaryAddQuotient = 0;
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    float epsilon = 0.0f;
    float momentum = 0.0f;
    float momentumReverse = 0.0f;
    int32_t useRunningMeanVar = 0;
};

struct BatchNorm3DInferTilingData {
    int64_t totalTiles = 0;
    int64_t tilesPerCore = 0;
    int64_t usedCoreNums = 0;
    int64_t totalB0Len = 0;
    int64_t totalALen = 0;
    int64_t totalB1Len = 0;
    int64_t b0Outer = 0;
    int64_t aOuter = 0;
    int64_t b1Outer = 0;
    int64_t tileBlockB0Len = 0;
    int64_t tileBlockB0Tail = 0;
    int64_t tileBlockALen = 0;
    int64_t tileBlockATail = 0;
    int64_t tileBlockB1Len = 0;
    int64_t tileBlockB1Tail = 0;
    int64_t tileBlockAPaddingNum = 0;
    float epsilon = 0.0f;
};

struct BatchNorm3DInferLastChannelTilingData {
    int64_t totalTiles = 0;
    int64_t tilesPerCore = 0;
    int64_t usedCoreNums = 0;
    int64_t totalALen = 0;
    int64_t aOuter = 0;
    int64_t bOuter = 0;
    int64_t tileBlockALen = 0;
    int64_t tileBlockATail = 0;
    int64_t tileBlockAPaddingNum = 0;
    int64_t tileBlockBLen = 0;
    int64_t tileBlockBTail = 0;
    float epsilon = 0.0f;
};

#endif // NORM_BATCH_NORM3_D_TILING_DATA_H
