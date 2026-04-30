/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_norm_tiling_def.h
 * \brief
 */

#ifndef __BATCH_NORM_TILING_DEF_H__
#define __BATCH_NORM_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X float_t
#define DTYPE_SCALE float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct BatchNormFullReduceRegbaseTilingData {
    int64_t r1;
    int64_t r0;
    int64_t a;
    int64_t aFactor;
    int64_t aBlockFactor;
    int64_t blockNum;
    int64_t r1r0LoopCount;
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;
    int64_t powerOfTwoForR;
    float epsilon;
    float momentum;
    int32_t useRunningMeanVar;
};

struct BatchNormRAFullReduceTilingData {
    int64_t r1;
    int64_t a;
    int64_t aFactor;
    int64_t aBlockFactor;
    int64_t blockNum;
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;
    int64_t powerOfTwoForR;
    float epsilon;
    float momentum;
    int32_t useRunningMeanVar;
};

struct BatchNormWelfordRegbaseTilingData {
    int64_t r1;
    int64_t r0;
    int64_t a0;
    int64_t loopR1outer;
    int64_t r1Factor;
    int64_t loopR0outer;
    int64_t r0Factor;
    int64_t realCoreNum;
    int64_t numLastCore;
    int64_t aBlockFactor;
    int64_t aGatherLimit;
    int64_t parallelN;
    int64_t processSize;
    int64_t ubSize;
    int64_t elemNum;
    int64_t vlLenFp32;
    int64_t cutR1OrR0;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;
    int64_t binaryAddQuotient;
    float epsilon;
    float momentum;
    int32_t useRunningMeanVar;
};

struct BatchNormRAWelfordTilingData {
    int64_t r;
    int64_t rFactor;
    int64_t a;
    int64_t aFactor;
    int64_t aBlockFactor;
    int64_t blockNum;
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;
    int64_t powerOfTwoForR;
    float epsilon;
    float momentum;
    int32_t useRunningMeanVar;
};

struct BatchNormBlockSplitRTilingData {
    int64_t patternR;
    int64_t patternA;
    int64_t patternAAlign;
    int64_t rUbFactor;
    int64_t tBufUbFactor;
    int64_t aUbFactor;
    int64_t aUbLoop;
    int64_t aUbTail;
    int64_t formerCoreBlockFactor;
    int64_t tailCoreBlockFactor;
    int64_t formerCoreNums;
    int64_t tailCoreNums;
    int64_t tailR;
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;
    int64_t lastBinaryAddQuotient;
    int64_t lastBinaryAddK;
    int64_t lastBinaryAddLast;
    float epsilon;
    float momentum;
    float momentumReverse;
    int32_t useRunningMeanVar;
};

struct BatchNormRARBlockSplitRTilingData {
    int64_t patternR1;
    int64_t patternA;
    int64_t patternR0;
    int64_t patternAAlign;
    int64_t blockSplitAxis;
    int64_t formerBlockOuter;
    int64_t tailBlockOuter;
    int64_t blockInner;
    int64_t ubFactor;
    int64_t formerCoreUbSplitAxis;
    int64_t formerCoreUbOuter;
    int64_t formerCoreUbInner;
    int64_t tailCoreUbSplitAxis;
    int64_t tailCoreUbOuter;
    int64_t tailCoreUbInner;
    int64_t formerCoreBinaryAddQuotient;
    int64_t tailCoreBinaryAddQuotient;
    int64_t lastBinaryAddQuotient;
    int64_t lastBinaryAddK;
    int64_t lastBinaryAddLast;
    float epsilon;
    float momentum;
    float momentumReverse;
    int32_t useRunningMeanVar;
};

struct BatchNormInferTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t totalB0Len;
    int64_t totalALen;
    int64_t totalB1Len;
    int64_t b0Outer;
    int64_t aOuter;
    int64_t b1Outer;
    int64_t tileBlockB0Len;
    int64_t tileBlockB0Tail;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockB1Len;
    int64_t tileBlockB1Tail;
    int64_t tileBlockAPaddingNum;
    float epsilon;
};

struct BatchNormInferLastChannelTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t totalALen;
    int64_t aOuter;
    int64_t bOuter;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockAPaddingNum;
    int64_t tileBlockBLen;
    int64_t tileBlockBTail;
    float epsilon;
};

#pragma pack()

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#endif