/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rms_norm_grad_quant_tiling_data.h
 * \brief
 */

#ifndef RMS_NORM_GRAD_QUANT_TILING_DATA_H
#define RMS_NORM_GRAD_QUANT_TILING_DATA_H

struct RmsNormGradQuantRegbaseDxTilingData {
    uint64_t rows;
    uint64_t cols;
    uint64_t blockFactorDx;
    uint64_t ubFactor;
    uint64_t bodyPart;
    uint64_t usedCoreNumDx;
};

struct RmsNormGradQuantRegbaseTilingData {
    uint64_t tailCoreNumDG;
    uint64_t colsPerCoreDG;
    uint64_t colsLastCoreDG;
    uint64_t colsPerTailCoreDG;
    uint64_t rowsPerUBDG;
    uint64_t powerofTwoValueDG;
    uint64_t rowsTailDG;
    uint64_t totalBlockCountDG;
    uint64_t mainBlockCountDG;
    uint64_t tailBlockCountwithPadDG;
    uint64_t powerOfTwoBlockCountDG;
    uint64_t tailBlockCountWithoutPadDG;
    uint64_t binaryAddKDG;
    uint64_t usedCoreNumDG;
    uint64_t blockSize;
    uint64_t vlFp32;
    uint64_t isFullLoad;
    uint64_t isMultiColset;

    RmsNormGradQuantRegbaseDxTilingData dxTilingData;
};

struct RmsNormGradQuantRegbaseBigMTilingData{
    uint64_t dgammaUsedCoreNum;
    uint64_t dgammaMPerBlock;
    uint64_t dgammaMReminder;
    uint64_t dgammaNloop;
    uint64_t dgammaNtail;
    uint64_t dgammaMfactorBlockAligned;
    uint64_t dgammaNfactorBlockAligned;
    uint64_t dgammaMToProcessMainBlock;
    uint64_t dgammaMLoopMainBlock;
    uint64_t dgammaMTotalLoopMainBlock;
    uint64_t dgammaMTailMainBlock;
    uint64_t dgammaBasicBlockLoopMainBlock;
    uint64_t dgammaMainFoldCountMainBlock;
    uint64_t dgammaCacheBufferCountMainBlock;
    uint64_t dgammaResultCacheIDMainBlock;
    uint64_t dgammaMToProcessTailBlock;
    uint64_t dgammaMLoopTailBlock;
    uint64_t dgammaMTotalLoopTailBlock;
    uint64_t dgammaMTailTailBlock;
    uint64_t dgammaBasicBlockLoopTailBlock;
    uint64_t dgammaMainFoldCountTailBlock;
    uint64_t dgammaCacheBufferCountTailBlock;
    uint64_t dgammaResultCacheIDTailBlock;
    uint64_t dgammaAInnerAlignedStg1;
    uint64_t dgammaAOuterStg1;
    uint64_t dgammaATailStg1;

    RmsNormGradQuantRegbaseDxTilingData dxTilingData;
};

struct RmsNormGradQuantRegbaseEmptyTilingData {
    uint64_t usedCoreNumDG;
    uint64_t colsPerCoreDG;
    uint64_t cols;
    uint32_t ubSize;
    uint64_t colsPerUBDG;
    uint64_t coreUbBlockCount;
    uint64_t tailUbCols;
    uint64_t lastCoreBlockCount;
    uint64_t lastCoreTailUbCols;
    uint64_t colsLastCoreDG;
};

#endif // RMS_NORM_GRAD_QUANT_TILING_DATA_H