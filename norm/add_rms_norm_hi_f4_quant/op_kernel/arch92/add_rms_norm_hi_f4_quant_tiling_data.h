/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ADD_RMS_NORM_HI_F4_QUANT_TILING_DATA_H_
#define _ADD_RMS_NORM_HI_F4_QUANT_TILING_DATA_H_

struct AddRmsNormHiF4QuantTilingData {
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t rowFactor;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
    uint64_t mxBlockSize;
    uint64_t blockNumInColAxis;
    uint64_t mxScaleSize;
    uint32_t betaFlag;
    uint32_t rstdFlag;
    int64_t scaleAlg;
    int64_t innerLoopEle;
    int64_t max8LoopEle;
    int64_t scaleLoopEle;
    int64_t outLoopEle;
};

struct AddRmsNormHiF4QuantReduceEmptyTilingData {
    uint64_t perCoreElements;
    uint64_t lastCoreElements;
    uint64_t perCoreLoops;
    uint64_t perCorePerLoopElements;
    uint64_t perCoreLastLoopElements;
    uint64_t lastCoreLoops;
    uint64_t lastCorePerLoopElements;
    uint64_t lastCoreLastLoopElements;
    uint64_t numRow;
    uint32_t rstdFlag;
};

struct AddRmsNormHiF4QuantSplitDTilingData {
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t mLastCore;
    uint64_t baseN;
    uint64_t baseNBlockSize;
    uint64_t baseM;
    uint64_t nUbLoops;
    uint64_t binAddQuotient;
    uint64_t powerSplit;
    uint64_t mainFoldCount;
    uint64_t foldTail;
    float epsilon;
    float avgFactor;
    uint64_t mxBlockSize;
    uint64_t mxScaleSize;
    uint32_t betaFlag;
    uint32_t rstdFlag;
    int64_t scaleAlg;
    int64_t innerLoopEle;
    int64_t max8LoopEle;
    int64_t scaleLoopEle;
    int64_t outLoopEle;
};

#endif
