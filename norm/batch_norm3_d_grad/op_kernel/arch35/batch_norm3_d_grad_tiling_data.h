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
 * \file batch_norm3_d_grad_tiling_data.h
 * \brief batch_norm3_d_grad plain tiling-data structs, shared by host tiling and kernel.
 */

#ifndef NORM_BATCH_NORM3_D_GRAD_TILING_DATA_H
#define NORM_BATCH_NORM3_D_GRAD_TILING_DATA_H

#include <cstdint>

struct BatchNorm3DGradBaseTilingData {
    int64_t r1Dim = 0;
    int64_t aDim = 0;
    int64_t r0Dim = 0;
    int64_t rAlign = 0;
    int64_t blockNum = 0;
    int64_t tailBlockNum = 0;
    int64_t formerBlockDim = 0;
    int64_t tailBlockDim = 0;
};

struct BatchNorm3DGradTilingData {
    int64_t dummy = 0;
};

struct BatchNorm3DGradBinaryAddTilingData {
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddk = 0;
    int64_t binaryAddLastNum = 0;
};

struct BatchNorm3DGradRARFullLoadTilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
    BatchNorm3DGradBinaryAddTilingData binaryAddTilingData;
    int64_t formerUbDim = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfTailBlock = 0;
};

struct BatchNorm3DGradRARRecomputeTilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
    BatchNorm3DGradBinaryAddTilingData generalBinAddTilingData;
    BatchNorm3DGradBinaryAddTilingData tailBinAddTilingData;
    int64_t ubRDimFactor = 0;
    int64_t ubRDimFactorAlign = 0;
    int64_t ubRDimLoopNum = 0;
    int64_t ubRDimTail = 0;
    int64_t ubRDimTailFactor = 0;
    int64_t ubRDimTailFactorAlign = 0;
    int64_t ubRDimTailLoopNum = 0;
    int64_t ubRDimTailTail = 0;
    int64_t ubRDimTailTailFactor = 0;
    int64_t ubRDimTailTailFactorAlign = 0;
    int64_t ubRDimTailTailLoopNum = 0;
};

struct BatchNorm3DGradRAFullLoadTilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
    BatchNorm3DGradBinaryAddTilingData binaryAddTilingData;
    int64_t numBlocks = 0;
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

struct BatchNorm3DGradRARecomputeTilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
    BatchNorm3DGradBinaryAddTilingData binaryAddTilingData;
    int64_t numBlocks = 0;
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
    float reciprocal = 0.0f;
};

struct BatchNorm3DGradRASplitRTilingData {
    int64_t rDim = 0;
    int64_t aDim = 0;
    int64_t usedCoreNum = 0;
    int64_t rLoopFactor = 0;
    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t binaryBlockCnt = 0;
    int64_t binaryFoldPoint = 0;
    int64_t binaryBlockTail = 0;
    int64_t lastCoreBlockCnt = 0;
    int64_t lastCoreFoldPoint = 0;
    int64_t lastCoreLoopTail = 0;
    int64_t aFactor = 0;
    int64_t aFactorAlign = 0;
    int64_t aFactorTail = 0;
    int64_t aLoopTimes = 0;
    int64_t dxLoopFactor = 0;
    int64_t dxLoopTail = 0;
    int64_t dxLoopTimes = 0;
    int64_t dxLastCoreFactor = 0;
    int64_t dxLastCoreTail = 0;
    int64_t dxLastCoreTimes = 0;
    int64_t cacheBuffCnt = 0;
};

struct BatchNorm3DGradInferChannelLastDxTilingData {
    int64_t totalTiles = 0;
    int64_t tilesPerCore = 0;
    int64_t usedCoreNums = 0;
    int64_t aDim = 0;
    int64_t aOuter = 0;
    int64_t bOuter = 0;
    int64_t tileBlockALen = 0;
    int64_t tileBlockATail = 0;
    int64_t tileBlockAPaddingNum = 0;
    int64_t tileBlockBLen = 0;
    int64_t tileBlockBTail = 0;
    float epsilon = 0.0f;
};

struct BatchNorm3DGradInferChannelLastTilingData {
    BatchNorm3DGradInferChannelLastDxTilingData dxTilingData;
    int64_t binAddRFactorStg1 = 0;
    int64_t binAddRLoopStg1 = 0;
    int64_t binAddRTotalLoopStg1 = 0;
    int64_t binAddRTailStg1 = 0;
    int64_t binAddBasicBlockLoopStg1 = 0;
    int64_t binAddMainFoldCountStg1 = 0;
    int64_t binAddCacheBufferCountStg1 = 0;
    int64_t binAddResultCacheIDStg1 = 0;
    int64_t aDimStg1 = 0;
    int64_t aOuterStg1 = 0;
    int64_t aInnerStg1 = 0;
    int64_t aTailStg1 = 0;
    int64_t aOuterPerCoreStg1 = 0;
    int64_t usedCoreNumsStg1 = 0;
    int32_t enableDx = 0;
    int32_t enableDgamma = 0;
    int32_t enableDbeta = 0;
};

struct BatchNorm3DGradInferDxTilingData {
    int64_t totalTiles = 0;
    int64_t tilesPerCore = 0;
    int64_t usedCoreNums = 0;
    int64_t r1Dim = 0;
    int64_t aDim = 0;
    int64_t r0Dim = 0;
    int64_t r1Outer = 0;
    int64_t aOuter = 0;
    int64_t r0Outer = 0;
    int64_t tileBlockR1Len = 0;
    int64_t tileBlockR1Tail = 0;
    int64_t tileBlockALen = 0;
    int64_t tileBlockATail = 0;
    int64_t tileBlockR0Len = 0;
    int64_t tileBlockR0Tail = 0;
    int64_t tileBlockAPaddingNum = 0;
    float epsilon = 0.0f;
};

struct BatchNorm3DGradInferTilingData {
    BatchNorm3DGradInferDxTilingData baseTilingData;
    BatchNorm3DGradBinaryAddTilingData generalBinAddTilingData;
    BatchNorm3DGradBinaryAddTilingData tailBinAddTilingData;
    int64_t blockNum = 0;
    int64_t tailBlockNum = 0;
    int64_t formerBlockDim = 0;
    int64_t tailBlockDim = 0;
    int64_t ubRDimFactor = 0;
    int64_t ubRDimFactorAlign = 0;
    int64_t ubRDimLoopNum = 0;
    int64_t ubRDimTail = 0;
    int64_t ubRDimTailFactor = 0;
    int64_t ubRDimTailFactorAlign = 0;
    int64_t ubRDimTailLoopNum = 0;
    int64_t ubRDimTailTail = 0;
    int64_t ubRDimTailTailFactor = 0;
    int64_t ubRDimTailTailFactorAlign = 0;
    int64_t ubRDimTailTailLoopNum = 0;
    int32_t enableDx = 0;
    int32_t enableDgamma = 0;
    int32_t enableDbeta = 0;
};

struct BatchNorm3DGradFullLoadTilingData {
    int64_t b1Dim = 0;
    int64_t aDim = 0;
    int64_t b0Dim = 0;
    int64_t bAlign = 0;
    int64_t coreChannelNum = 0;
    int64_t coreChannelNumTail = 0;
    int64_t cUbBlock = 0;
    int64_t needCoreNum = 0;
    float epsilon = 0.0f;
};

struct BatchNorm3DGradRARSplitCoreR1TilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
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
    int64_t r0TailAlignedStg0 = 0;
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
    int64_t r0TailAlignedStg2 = 0;
    int64_t r1InnerInnerStg2 = 0;
    int64_t r1InnerOuterStg2 = 0;
    int64_t r1InnerTailStg2 = 0;
    int64_t r1TailOuterStg2 = 0;
    int64_t r1TailTailStg2 = 0;
    int64_t aInnerStg2 = 0;
    int64_t aInnerAlignedStg2 = 0;
    int64_t aOuterStg2 = 0;
    int64_t aTailStg2 = 0;
    int64_t binAddBasicBlockLoop = 0;
    int64_t binAddMainFoldCount = 0;
    int64_t binAddCacheBufferCount = 0;
    int64_t binAddResultCacheID = 0;
    int64_t lastCoreBinAddBasicBlockLoop = 0;
    int64_t lastCoreBinAddMainFoldCount = 0;
    int64_t lastCoreBinAddResultCacheID = 0;
};

struct BatchNorm3DGradRARSplitCoreR0TilingData {
    BatchNorm3DGradBaseTilingData baseTilingData;
    int64_t r1Dim = 0;
    int64_t aDim = 0;
    int64_t aDimAligned = 0;
    int64_t r0Dim = 0;
    int64_t usedCoreNums = 0;
    int64_t r0Inner = 0;
    int64_t r0Tail = 0;
    int64_t r0InnerInnerStg0 = 0;
    int64_t r0InnerOuterStg0 = 0;
    int64_t r0InnerTailStg0 = 0;
    int64_t r0TailOuterStg0 = 0;
    int64_t r0TailTailStg0 = 0;
    int64_t r0TailTailAlignedStg0 = 0;
    int64_t r1InnerStg0 = 0;
    int64_t r1OuterStg0 = 0;
    int64_t r1TailStg0 = 0;
    int64_t aInnerStg0 = 0;
    int64_t aInnerAlignedStg0 = 0;
    int64_t aOuterStg0 = 0;
    int64_t aTailStg0 = 0;
    int64_t aInnerStg1 = 0;
    int64_t aOuterStg1 = 0;
    int64_t aTailStg1 = 0;
    int64_t r1InnerStg2 = 0;
    int64_t r1OuterStg2 = 0;
    int64_t r1TailStg2 = 0;
    int64_t r0InnerInnerStg2 = 0;
    int64_t r0InnerOuterStg2 = 0;
    int64_t r0InnerTailStg2 = 0;
    int64_t r0TailOuterStg2 = 0;
    int64_t r0TailTailStg2 = 0;
    int64_t r0TailTailAlignedStg2 = 0;
    int64_t aInnerStg2 = 0;
    int64_t aInnerAlignedStg2 = 0;
    int64_t aOuterStg2 = 0;
    int64_t aTailStg2 = 0;
    int64_t binAddBasicBlockLoop = 0;
    int64_t binAddMainFoldCount = 0;
    int64_t binAddCacheBufferCount = 0;
    int64_t binAddResultCacheID = 0;
    int64_t lastCoreBinAddBasicBlockLoop = 0;
    int64_t lastCoreBinAddMainFoldCount = 0;
    int64_t lastCoreBinAddResultCacheID = 0;
};

#endif // NORM_BATCH_NORM3_D_GRAD_TILING_DATA_H
