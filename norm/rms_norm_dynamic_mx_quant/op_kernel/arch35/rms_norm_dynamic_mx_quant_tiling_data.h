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
 * \file rms_norm_dynamic_mx_quant_tiling_data.h
 * \brief RmsNormDynamicMxQuant tiling data structures for kernel
 */

#ifndef __OP_KERNEL_RMS_NORM_DYNAMIC_MX_QUANT_TILING_DATA_H__
#define __OP_KERNEL_RMS_NORM_DYNAMIC_MX_QUANT_TILING_DATA_H__

#ifndef __CCE_AICORE__
#include <cstdint>
#else
#include "kernel_tiling/kernel_tiling.h"
#endif

#pragma pack(push, 8)
struct alignas(8) RmsNormDynamicMxQuantFullLoadTilingData {
    int64_t usedCoreNum = 0;
    int64_t mTailCores = 0;
    int64_t numM = 0;
    int64_t numN = 0;
    int64_t binAddFoldPoint = 0;
    int64_t mPerCore = 0;
    int64_t mUbFactor = 0;
    int64_t mxBlockSize = 0;
    int64_t nMxblockAligned = 0;
    int64_t nMxblockNumAlignedTwo = 0;
    int64_t needPadN = 0;
    int64_t scaleAlg = 0;
    int64_t roundMode = 0;
    int64_t hasInputBeta = 0;
    int64_t hasOutputRstd = 0;
    float epsilon = 0.0f;
    float avgFactor = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) RmsNormDynamicMxQuantRecomputeTilingData {
    int64_t usedCoreNum = 0;
    int64_t mPerCore = 0;
    int64_t mTailCores = 0;
    int64_t numM = 0;
    int64_t numN = 0;
    int64_t baseM = 0;
    int64_t baseN = 0;
    int64_t baseNTail = 0;
    int64_t baseNTailAligned = 0;
    int64_t nUbLoops = 0;
    int64_t binAddQuotient = 0;
    int64_t powerSplit = 0;
    int64_t mainFoldCount = 0;
    int64_t foldTail = 0;
    int64_t resultCacheId = 0;
    int64_t mxBlockSize = 0;
    int64_t nMxblockNumAlignedTwo = 0;
    int64_t needPadN = 0;
    int64_t scaleAlg = 0;
    int64_t roundMode = 0;
    int64_t hasInputBeta = 0;
    int64_t hasOutputRstd = 0;
    float epsilon = 0.0f;
    float avgFactor = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) RmsNormDynamicMxQuantReduceEmptyTilingData {
    uint64_t perCoreElements = 0;
    uint64_t lastCoreElements = 0;
    uint64_t perCoreLoops = 0;
    uint64_t perCorePerLoopElements = 0;
    uint64_t perCoreLastLoopElements = 0;
    uint64_t lastCoreLoops = 0;
    uint64_t lastCorePerLoopElements = 0;
    uint64_t lastCoreLastLoopElements = 0;
    uint64_t hasOutputRstd = 0;
    uint64_t numM = 0;
};
#pragma pack(pop)

#endif // __OP_KERNEL_RMS_NORM_DYNAMIC_MX_QUANT_TILING_DATA_H__