/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_DATA_H
#define SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_TILING_DATA_H
#include <cstdint>

// Plain Kernel layouts; kept byte-compatible with the Host serialization.
struct alignas(8) SwigluGroupQuantWithDualAxisTilingData {
    uint32_t version;
    uint32_t quantMode;
    uint32_t inputType;
    uint32_t weightType;
    uint32_t flags;
    int64_t t;
    int64_t h;
    int64_t keep;
    int64_t groupCount;
    int64_t batchRows;
    float alpha;
    float bias;
    float clampLimit;
    int64_t scale1RowBytes;
    int64_t scale2PairRows;
    int64_t rowOfFormerBlock;
    int64_t rowOfTailBlock;
    int64_t rowLoopOfFormerBlock;
    int64_t rowLoopOfTailBlock;
    int64_t rowFactor;
    int64_t tailRowFactorOfFormerBlock;
    int64_t tailRowFactorOfTailBlock;
    int64_t dLoop;
    int64_t dFactor;
    int64_t tailDFactor;
    int64_t usedCoreCount;
    int64_t tileRows;
    int64_t tileCols;
    int64_t groupChunkSize;
    uint64_t ubUserBytes;
    uint64_t ubPlannedBytes;
};

#endif
