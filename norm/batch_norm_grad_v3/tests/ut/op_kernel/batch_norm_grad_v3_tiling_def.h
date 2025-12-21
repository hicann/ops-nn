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
 * \file batch_norm_grad_v3_tiling_def.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_TILING_H__
#define __BATCH_NORM_GRAD_V3_TILING_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_DY float_t
#define DTYPE_WEIGHT float_t
#define DTYPE_RUNNING_VAR float_t
#define DTYPE_SAVE_MEAN float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct BatchNormGradV3BaseTilingData {
    int64_t r1Dim;
    int64_t aDim;
    int64_t r0Dim;
    int64_t rAlign;
    int64_t blockNum;
    int64_t tailBlockNum;
    int64_t formerBlockDim;
    int64_t tailBlockDim;
};

struct BatchNormGradV3BinaryAddTilingData {
    int64_t binaryAddQuotient;
    int64_t binaryAddk;
    int64_t binaryAddLastNum;
};

struct BatchNormGradV3RARFullLoadTilingData {
    BatchNormGradV3BaseTilingData baseTilingData;
    BatchNormGradV3BinaryAddTilingData binaryAddTilingData;
    int64_t formerUbDim = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;
};

struct BatchNormGradV3RARRecomputeTilingData {
    BatchNormGradV3BaseTilingData baseTilingData;
    BatchNormGradV3BinaryAddTilingData generalBinAddTilingData;
    BatchNormGradV3BinaryAddTilingData tailBinAddTilingData;
    int64_t ubRDimFactor = 0;              // 一次完整的UB内，循环执行R轴的个数
    int64_t ubRDimFactorAlign = 0;         // 一次完整的UB内，循环对齐到block的R轴大小
    int64_t ubRDimLoopNum = 0;             // 核内R轴UB循环的次数
    int64_t ubRDimTail = 0;                // R轴尾块大小，R - ubRDimFactor * ubRDimLoopNum
    int64_t ubRDimTailFactor = 0;          // R轴尾块一次循环处理的R轴个数
    int64_t ubRDimTailFactorAlign = 0;     // R轴尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailLoopNum = 0;         // R轴尾块循环次数
    int64_t ubRDimTailTail = 0;            // R轴尾块的尾块的个数
    int64_t ubRDimTailTailFactor = 0;      // R轴尾块的尾块一次循环处理的R轴个数
    int64_t ubRDimTailTailFactorAlign = 0; // R轴尾块的尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailTailLoopNum = 0;     // R轴尾块的尾块的循环次数
};

struct BatchNormGradV3RAFullLoadTilingData {
    BatchNormGradV3BaseTilingData baseTilingData;
    BatchNormGradV3BinaryAddTilingData binaryAddTilingData;
    int64_t blockDim = 0;
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t mainBlockCount = 0;
    int64_t tailBlockCount = 0;
    int64_t mainALoopFactor = 0;
    int64_t mainALoopFactorAligned = 0;
    int64_t tailALoopFactor = 0;
    int64_t tailALoopFactorAligned = 0;
    int64_t foldLoopStep1 = 0;
    int64_t foldLoopOffset1 = 0;
    int64_t foldLoopStep2 = 0;
    int64_t foldLoopOffset2 = 0;
    int64_t foldLoopStep3 = 0;
    int64_t foldLoopOffset3 = 0;
    int64_t reduceRecursionLoop = 0;
    int64_t reduceLoopTimes = 0;
};

struct BatchNormGradV3RARecomputeTilingData {
    BatchNormGradV3BaseTilingData baseTilingData;
    BatchNormGradV3BinaryAddTilingData binaryAddTilingData;
    int64_t blockDim = 0;
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t mainBlockCount = 0;
    int64_t tailBlockCount = 0;
    int64_t aLoopFactor = 0;
    int64_t aLoopFactorAligned = 0;
    int64_t rLoopFactor = 0;
    int64_t rLoopTimes = 0;
    int64_t rLoopTail = 0;
    int64_t binaryFoldPoint = 0;
    int64_t binaryBlockCount = 0;
    int64_t binaryTailBlock = 0;
    int64_t cacheBufferCount = 0;
    float reciprocal = 0;
};

struct BatchNormGradV3RARSplitCoreR1TilingData {
    BatchNormGradV3BaseTilingData baseTilingData;
    int64_t r1Dim = 0;
    int64_t aDim = 0;
    int64_t aDimAligned = 0;
    int64_t r0Dim = 0;
    int64_t usedCoreNums = 0;
    int64_t r1Inner = 0;
    int64_t r1Tail = 0;
    int64_t r0InnerStg0 = 0;
    int64_t r0OuterStg0 = 0;
    int64_t r0TailStg0 = 0;
    int64_t r1InnerInnerStg0 = 0;
    int64_t r1InnerOuterStg0 = 0;
    int64_t r1InnerTailStg0 = 0;
    int64_t r1TailOuterStg0 = 0;
    int64_t r1TailTailStg0 = 0;
    int64_t aInnerStg0 = 0;
    int64_t aInnerAlignedStg0 = 0;
    int64_t aOuterStg0 = 0;
    int64_t aTailStg0 = 0;
    int64_t aInnerStg1 = 0;
    int64_t aOuterStg1 = 0;
    int64_t aTailStg1 = 0;
    int64_t r0InnerStg2 = 0;
    int64_t r0OuterStg2 = 0;
    int64_t r0TailStg2 = 0;
    int64_t r1InnerInnerStg2 = 0;
    int64_t r1InnerOuterStg2 = 0;
    int64_t r1InnerTailStg2 = 0;
    int64_t r1TailOuterStg2 = 0;
    int64_t r1TailTailStg2 = 0;
    int64_t aInnerStg2 = 0;
    int64_t aInnerAlignedStg2 = 0;
    int64_t aOuterStg2 = 0;
    int64_t aTailStg2 = 0;
};

struct BatchNormGradV3InferChannelLastTilingData {
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
    float epsilon = 0;
};

struct BatchNormGradV3InferTilingData {
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
    float epsilon = 0;
};

struct BatchNormGradV3FullLoadTilingData {
    int64_t b1Dim = 0;
    int64_t aDim = 0;
    int64_t b0Dim = 0;
    int64_t bAlign = 0;
    int64_t coreChannelNum = 0;
    int64_t coreChannelNumTail = 0;
    int64_t cUbBlock = 0;
    int64_t needCoreNum = 0;
    float epsilon = 0;
};

struct BatchNormGradV3SplitLoadTilingData {
    int64_t b1Dim = 0;
    int64_t aDim = 0;
    int64_t b0Dim = 0;
    int64_t bAlign = 0;
    int64_t coreChannelNum = 0;
    int64_t coreChannelNumTail = 0;
    int64_t bUbBlock = 0;
    int64_t b0UbBlock = 0;
    int64_t bUbLoop = 0;
    int64_t bUbBlockTail = 0;
    int64_t b0UbBlockTail = 0;
    int64_t needCoreNum = 0;
    float epsilon = 0;
};

struct BatchNormGradV3SplitLoadCrossCoreTilingData {
    int64_t b1Dim = 0;
    int64_t aDim = 0;
    int64_t b0Dim = 0;
    int64_t bAlign = 0;
    int64_t coreChannelNum = 0;
    int64_t coreChannelNumTail = 0;
    int64_t bUbBlock = 0;
    int64_t b0UbBlock = 0;
    int64_t bUbLoop = 0;
    int64_t bUbTailLoop = 0;
    int64_t bUbBlockTail = 0;
    int64_t b0UbBlockTail = 0;
    int64_t b0UbTailBlockTail = 0;
    int64_t needCoreNum = 0;
    int64_t groupCore = 0;
    int64_t groupBlockNum = 0;
    int64_t groupBlockNumTail = 0;
    int64_t morehalfChannel = 0;
    int64_t groupBlockNumHalf = 0;
    int64_t groupBlockNumHalfTail = 0;
    int64_t bUbLoopHalf = 0;
    int64_t b0UbBlockTailHalf = 0;
    int64_t bUbTailLoopHalf = 0;
    int64_t b0UbTailBlockTailHalf = 0;
    int64_t moreMultiChannel = 0;
    int64_t bUbLoopMulti = 0;
    int64_t b0UbBlockTailMulti = 0;
    int64_t coreNum = 0;
    float epsilon = 0;
};

#pragma pack()

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#endif