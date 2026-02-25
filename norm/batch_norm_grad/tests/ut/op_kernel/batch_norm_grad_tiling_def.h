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
struct BatchNormGradBaseTilingData {
    int64_t r1Dim;
    int64_t aDim;
    int64_t r0Dim;
    int64_t rAlign;
    int64_t blockNum;
    int64_t tailBlockNum;
    int64_t formerBlockDim;
    int64_t tailBlockDim;
};

struct BatchNormGradTilingData {
    int64_t dummy;
};

struct BatchNormGradBinaryAddTilingData {
    int64_t binaryAddQuotient;
    int64_t binaryAddk;
    int64_t binaryAddLastNum;
};

struct BatchNormGradRARFullLoadTilingData {
    BatchNormGradBaseTilingData baseTilingData;
    BatchNormGradBinaryAddTilingData binaryAddTilingData;
    int64_t formerUbDim;         // 一次完整的UB内，循环执行A轴的次数
    int64_t ubLoopOfFormerBlock; // 整核进行UB循环的次数，formerBlockDim/formerUbDim 向上取整
    int64_t ubTailOfFormerBlock; // 整核最后一轮UB循环，处理的A轴个数
    int64_t ubLoopOfTailBlock; // 尾核进行UB循环的次数，tailBlockDim/formerUbDim 向上取整
    int64_t ubTailOfTailBlock; // 尾核最后一轮UB循环，处理的A轴个数
};

struct BatchNormGradRARRecomputeTilingData {
    BatchNormGradBaseTilingData baseTilingData;
    BatchNormGradBinaryAddTilingData generalBinAddTilingData; // 整块的二分累加参数
    BatchNormGradBinaryAddTilingData tailBinAddTilingData; // 整块+尾块的二分累加参数
    int64_t ubRDimFactor;              // 一次完整的UB内，循环执行R轴的个数
    int64_t ubRDimFactorAlign;         // 一次完整的UB内，循环对齐到block的R轴大小
    int64_t ubRDimLoopNum;             // 核内R轴UB循环的次数
    int64_t ubRDimTail;                // R轴尾块大小，R - ubRDimFactor * ubRDimLoopNum
    int64_t ubRDimTailFactor;          // R轴尾块一次循环处理的R轴个数
    int64_t ubRDimTailFactorAlign;     // R轴尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailLoopNum;         // R轴尾块循环次数
    int64_t ubRDimTailTail;            // R轴尾块的尾块的个数
    int64_t ubRDimTailTailFactor;      // R轴尾块的尾块一次循环处理的R轴个数
    int64_t ubRDimTailTailFactorAlign; // R轴尾块的尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailTailLoopNum;     // R轴尾块的尾块的循环次数
};

struct BatchNormGradRAFullLoadTilingData {
    BatchNormGradBaseTilingData baseTilingData;
    BatchNormGradBinaryAddTilingData binaryAddTilingData;
    int64_t numBlocks;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;
    int64_t mainBlockCount;
    int64_t tailBlockCount;
    int64_t mainALoopFactor;
    int64_t mainALoopFactorAligned;
    int64_t tailALoopFactor;
    int64_t tailALoopFactorAligned;
    int64_t foldLoopStep1;
    int64_t foldLoopOffset1;
    int64_t foldLoopStep2;
    int64_t foldLoopOffset2;
    int64_t foldLoopStep3;
    int64_t foldLoopOffset3;
    int64_t reduceRecursionLoop;
    int64_t reduceLoopTimes;
};

struct BatchNormGradRARecomputeTilingData {
    BatchNormGradBaseTilingData baseTilingData;
    BatchNormGradBinaryAddTilingData binaryAddTilingData;
    int64_t numBlocks;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;
    int64_t mainBlockCount;
    int64_t tailBlockCount;
    int64_t aLoopFactor;
    int64_t aLoopFactorAligned;
    int64_t rLoopFactor;
    int64_t rLoopTimes;
    int64_t rLoopTail;
    int64_t binaryFoldPoint;
    int64_t binaryBlockCount;
    int64_t binaryTailBlock;
    int64_t cacheBufferCount;
    float reciprocal;
};

struct BatchNormGradRASplitRTilingData {
    int64_t rDim;
    int64_t aDim;
    int64_t usedCoreNum;       // 需要多少个核
    int64_t rLoopFactor;       // 一次循环处理多少个R
    int64_t blockFactor;       // 每个核处理多少个R
    int64_t tailBlockFactor;   // 尾核处理多少个R
    int64_t binaryBlockCnt;    // ub内rLoopFactor需要循环几次
    int64_t binaryFoldPoint;   // ub内二分折叠点
    int64_t binaryBlockTail;   // 最后一次处理多少个R
    int64_t lastCoreBlockCnt;  // 尾核ub内rLoopFactor需要循环几次
    int64_t lastCoreFoldPoint; // 尾核ub内二分折叠点
    int64_t lastCoreLoopTail;  // 尾核最后一次处理多少个R
    int64_t aFactor;           // 一次循环可以A轴处理多少
    int64_t aFactorAlign;      // a方向对齐大小
    int64_t aFactorTail;       // 最后一次处理多少
    int64_t aLoopTimes;        // A轴需要处理多少次
    int64_t dxLoopFactor;      // 计算dx时一次处理多少行
    int64_t dxLoopTail;        // 计算dx时尾块处理多少行
    int64_t dxLoopTimes;       // 计算dx时R轴需要循环多少次
    int64_t dxLastCoreFactor;  // 计算dx时尾核一次处理多少行
    int64_t dxLastCoreTail;    // 计算dx时尾核尾块处理多少行
    int64_t dxLastCoreTimes;   // 计算dx时尾核R轴需要循环多少次
    int64_t cacheBuffCnt;      // cacheBuffer个数
};

// inference
struct BatchNormGradInferChannelLastDxTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t aDim;
    int64_t aOuter;
    int64_t bOuter;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockAPaddingNum;
    int64_t tileBlockBLen;
    int64_t tileBlockBTail;
    float epsilon;
};

struct BatchNormGradInferChannelLastTilingData {
    BatchNormGradInferChannelLastDxTilingData dxTilingData;
    // BatchNormGradBaseTilingData baseTilingData;
    // BatchNormGradBinaryAddTilingData binaryAddTilingData;
    int64_t binAddRFactorStg1;
    int64_t binAddRLoopStg1;
    int64_t binAddRTotalLoopStg1;
    int64_t binAddRTailStg1;
    int64_t binAddBasicBlockLoopStg1;
    int64_t binAddMainFoldCountStg1;
    int64_t binAddCacheBufferCountStg1;
    int64_t binAddResultCacheIDStg1;
    int64_t aDimStg1;
    int64_t aOuterStg1;
    int64_t aInnerStg1;
    int64_t aTailStg1;
    int64_t aOuterPerCoreStg1;
    int64_t usedCoreNumsStg1;
    int32_t enableDx;
    int32_t enableDgamma;
    int32_t enableDbeta;
};

struct BatchNormGradInferDxTilingData {
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t r1Dim;
    int64_t aDim;
    int64_t r0Dim;
    int64_t r1Outer;
    int64_t aOuter;
    int64_t r0Outer;
    int64_t tileBlockR1Len;
    int64_t tileBlockR1Tail;
    int64_t tileBlockALen;
    int64_t tileBlockATail;
    int64_t tileBlockR0Len;
    int64_t tileBlockR0Tail;
    int64_t tileBlockAPaddingNum;
    float epsilon;
};

struct BatchNormGradInferTilingData {
    BatchNormGradInferDxTilingData baseTilingData;
    BatchNormGradBinaryAddTilingData generalBinAddTilingData; // 整块的二分累加参数
    BatchNormGradBinaryAddTilingData tailBinAddTilingData; // 整块+尾块的二分累加参数
    int64_t blockNum;       // 参与计算的所有核数
    int64_t tailBlockNum;   // 尾核的个数
    int64_t formerBlockDim; // 整核分配的A轴个数，aDim/blockNum 向下取整
    int64_t tailBlockDim;   // 尾核分配的A轴个数，aDim/blockNum 向上取整
    int64_t ubRDimFactor;              // 一次完整的UB内，循环执行R轴的个数
    int64_t ubRDimFactorAlign;         // 一次完整的UB内，循环对齐到block的R轴大小
    int64_t ubRDimLoopNum;             // 核内R轴UB循环的次数
    int64_t ubRDimTail;                // R轴尾块大小，R - ubRDimFactor * ubRDimLoopNum
    int64_t ubRDimTailFactor;          // R轴尾块一次循环处理的R轴个数
    int64_t ubRDimTailFactorAlign;     // R轴尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailLoopNum;         // R轴尾块循环次数
    int64_t ubRDimTailTail;            // R轴尾块的尾块的个数
    int64_t ubRDimTailTailFactor;      // R轴尾块的尾块一次循环处理的R轴个数
    int64_t ubRDimTailTailFactorAlign; // R轴尾块的尾块一次循环对齐到block的R轴大小
    int64_t ubRDimTailTailLoopNum;     // R轴尾块的尾块的循环次数
    int32_t enableDx;
    int32_t enableDgamma;
    int32_t enableDbeta;
};

struct BatchNormGradFullLoadTilingData {
    int64_t b1Dim;
    int64_t aDim;
    int64_t b0Dim;
    int64_t bAlign;
    int64_t coreChannelNum;
    int64_t coreChannelNumTail;
    int64_t cUbBlock;
    int64_t needCoreNum;
    float epsilon;
};

struct BatchNormGradRARSplitCoreR1TilingData {
    BatchNormGradBaseTilingData baseTilingData;
    int64_t r1Dim;
    int64_t aDim;
    int64_t aDimAligned;
    int64_t r0Dim;
    int64_t usedCoreNums;
    int64_t r1Inner;
    int64_t r1Tail;
    int64_t r0InnerStg0;
    int64_t r0OuterStg0;
    int64_t r0TailStg0;
    int64_t r0TailAlignedStg0;
    int64_t r1InnerInnerStg0;
    int64_t r1InnerOuterStg0;
    int64_t r1InnerTailStg0;
    int64_t r1TailOuterStg0;
    int64_t r1TailTailStg0;
    int64_t aInnerStg0;
    int64_t aInnerAlignedStg0;
    int64_t aOuterStg0;
    int64_t aTailStg0;
    int64_t aInnerStg1;
    int64_t aOuterStg1;
    int64_t aTailStg1;
    int64_t r0InnerStg2;
    int64_t r0OuterStg2;
    int64_t r0TailStg2;
    int64_t r0TailAlignedStg2;
    int64_t r1InnerInnerStg2;
    int64_t r1InnerOuterStg2;
    int64_t r1InnerTailStg2;
    int64_t r1TailOuterStg2;
    int64_t r1TailTailStg2;
    int64_t aInnerStg2;
    int64_t aInnerAlignedStg2;
    int64_t aOuterStg2;
    int64_t aTailStg2;
    int64_t binAddBasicBlockLoop;
    int64_t binAddMainFoldCount;
    int64_t binAddCacheBufferCount;
    int64_t binAddResultCacheID;
    int64_t lastCoreBinAddBasicBlockLoop;
    int64_t lastCoreBinAddMainFoldCount;
    int64_t lastCoreBinAddResultCacheID;
};

struct BatchNormGradRARSplitCoreR0TilingData {
    BatchNormGradBaseTilingData baseTilingData;
    int64_t r1Dim;
    int64_t aDim;
    int64_t aDimAligned;
    int64_t r0Dim;
    int64_t usedCoreNums;
    int64_t r0Inner;
    int64_t r0Tail;
    int64_t r0InnerInnerStg0;
    int64_t r0InnerOuterStg0;
    int64_t r0InnerTailStg0;
    int64_t r0TailOuterStg0;
    int64_t r0TailTailStg0;
    int64_t r0TailTailAlignedStg0;
    int64_t r1InnerStg0;
    int64_t r1OuterStg0;
    int64_t r1TailStg0;
    int64_t aInnerStg0;
    int64_t aInnerAlignedStg0;
    int64_t aOuterStg0;
    int64_t aTailStg0;
    int64_t aInnerStg1;
    int64_t aOuterStg1;
    int64_t aTailStg1;
    int64_t r1InnerStg2;
    int64_t r1OuterStg2;
    int64_t r1TailStg2;
    int64_t r0InnerInnerStg2;
    int64_t r0InnerOuterStg2;
    int64_t r0InnerTailStg2;
    int64_t r0TailOuterStg2;
    int64_t r0TailTailStg2;
    int64_t r0TailTailAlignedStg2;
    int64_t aInnerStg2;
    int64_t aInnerAlignedStg2;
    int64_t aOuterStg2;
    int64_t aTailStg2;
    int64_t binAddBasicBlockLoop;
    int64_t binAddMainFoldCount;
    int64_t binAddCacheBufferCount;
    int64_t binAddResultCacheID;
    int64_t lastCoreBinAddBasicBlockLoop;
    int64_t lastCoreBinAddMainFoldCount;
    int64_t lastCoreBinAddResultCacheID;
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