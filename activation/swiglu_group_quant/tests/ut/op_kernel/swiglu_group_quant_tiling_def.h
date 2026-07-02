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
 * \file swiglu_group_quant_tiling_def.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_DEF_H_
#define SWIGLU_GROUP_QUANT_TILING_DEF_H_

#include <cstring>
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

struct SwigluGroupQuantTilingData {
    int64_t bs;
    int64_t d;
    int64_t splitD;
    int64_t scaleCol;
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
    int64_t roundScale;
    int64_t outputOrigin;
    float clampLimit;
    int64_t hasClampLimit;
    int64_t g;
    int64_t ubSize;
    int64_t gLoop;
    int64_t gFactor;
    int64_t tailGFactor;
    int64_t coreNum;
};

struct SwigluGroupQuantHifp8TilingData {
    int64_t totalTokens;
    int64_t dim2H;
    int64_t dimH;
    int64_t isGroup;
    int64_t hasWeight;
    int64_t hasClamp;
    int64_t outputOrigin;
    float clampLimit;
    float dstTypeMax;
    int64_t tileTokens;
    int64_t usedCoreNum;
    int64_t tokensPerCore;
    int64_t groupNum;
    int64_t tileLength;
};

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(T));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    SwigluGroupQuantTilingData tilingData;     \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif // SWIGLU_GROUP_QUANT_TILING_DEF_H_
