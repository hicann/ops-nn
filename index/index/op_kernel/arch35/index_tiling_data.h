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
 * \file index_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _INDEX_TILING_DATA_H_
#define _INDEX_TILING_DATA_H_

namespace Index {

struct IndexSimtTilingData {
    uint64_t inputLength = 0;
    uint64_t outputLength = 0;
    uint64_t indexSize = 0;
    uint32_t inputDimNum = 0;
    uint32_t indexedDimNum = 0;
    uint32_t indexedSizesNum = 0;
    uint32_t accumulateMode = 0;
    uint64_t inputShape[8] = {0};
};

struct IndexPerfSimtTilingData {
    uint64_t outputLength = 0;
    uint64_t inputShape0 = 0;
    uint64_t inputShape1 = 0;
};

struct IndexFullLoadTilingData {
    uint32_t inputStride[8] = {0};
    uint32_t inputShape[8] = {0};
    uint32_t indicesNum = 0;
    uint32_t usedCoreNum = 0;
    int64_t eachCoreSize = 0;
    int64_t normalCoreLoop = 0;
    int64_t normalCoreTail = 0;
    int64_t tailCoreLoop = 0;
    int64_t tailCoreTail = 0;
    int64_t lastDimSize = 0;
    int64_t inputQueSize = 0;
    int64_t indicesQueSize = 0;
    int64_t outputQueSize = 0;
    int64_t ubIndices = 0;
    int64_t inputShapeSize = 0;
};

struct IndexNonContinuousTilingData {
    uint64_t inputLength = 0;
    uint64_t outputLength = 0;
    uint64_t indexSize = 0;
    uint32_t inputDimNum = 0;
    uint32_t indexedDimNum = 0;
    uint32_t indexedSizesNum = 0;
    uint32_t accumulateMode = 0;
    uint64_t valueDimNum = 0;
    int64_t xShape[4] = {0};
    int64_t indexShape[4] = {0};
    int64_t valueShape[8] = {0};
    int64_t xStride[4] = {0};
    int64_t indexStride1[4] = {0};
    int64_t indexStride2[4] = {0};
    int64_t indexStride3[4] = {0};
    int64_t indexStride4[4] = {0};
    int64_t valueStride[8] = {0};
    int64_t yStride[4] = {0};
};

struct IndexSimdTilingData {
    uint32_t indexedDimNum = 0;
    uint64_t mergeInputShape[8] = {0};
    uint32_t mergeInputIndexed[8] = {0};
    uint32_t mergeInputShapeDim = 0;
    uint64_t mergeOutputShape[8] = {0};
    int32_t mergeOutToInput[8] = {0};
    int32_t indicesToInput[8] = {0};
    uint32_t mergeOutputShapeDim = 0;
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t maxElement = 0;
    int64_t indiceUbSize = 0;
    int64_t inputDtypeSize = 0;
    uint64_t indexSize = 0;
    int32_t isZeroOneZero = 0;
};

} // namespace Index

#endif
