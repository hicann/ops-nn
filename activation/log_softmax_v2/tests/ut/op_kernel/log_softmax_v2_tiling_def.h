/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file log_softmax_v2_tiling_def.h
 * \brief
 */

#ifndef __LOG_SOFTMAX_V2_TILING_DEF_H__
#define __LOG_SOFTMAX_V2_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_LOGITS float_t
#define DTYPE_LOGSOFTMAX float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct SoftmaxV2ArSmallRTilingData {
    int64_t totalA0Len;
    int64_t totalRLen;
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t tileA0Len;
    int64_t tileA0Tail;
    int64_t rTileBase;
    int64_t rAligned;
};

struct SoftmaxV2ARTilingData {
    int64_t a;
    int64_t r;
    int64_t rAligned;
    int64_t ubFactor;
    int64_t aBlockFactor;
    int64_t rLoopCount;
};

struct SoftmaxV2ArRecomputeTilingData {
    int64_t a;
    int64_t r;
    int64_t ubFactor;
    int64_t ubFactorTail;
    int64_t aBlockFactor;
    int64_t aLoopCountCeil;
    int64_t basicBlockLoop;
    int64_t mainFoldCount;
};

struct SoftmaxV2ARATilingData {
    int64_t totalA1Len;
    int64_t totalRLen;
    int64_t totalA0Len;
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t a1Outer;
    int64_t tileA1Len;
    int64_t tileA1Tail;
    int64_t a0Outer;
    int64_t tileA0Len;
    int64_t tileA0Tail;
    uint16_t binaryAddK;
    uint16_t binaryAddLast;
    uint16_t binaryAddInnerLoop;
    uint16_t remainderLoopCount;
    uint16_t quotientLoopCount;
    uint16_t remainderTailCount;
    uint32_t remainderOffset;
    uint32_t baseLineOffset;
    uint32_t validNumInXUb;
    uint32_t quotientTailOffset;
    uint32_t remainderTailOffset;
    uint32_t remainderTailOffset0;
    uint32_t remainderTailOffset1;
    uint32_t remainderTailOffset2;
    uint32_t remainderTailOffset3;
    uint32_t remainderTailOffset4;
    uint32_t remainderTailOffset5;
    uint32_t remainderTailOffset6;
    uint32_t remainderTailOffset7;
};

struct SoftmaxV2ARARecomputeTilingData {
    int64_t totalRLen;
    int64_t totalA0Len;
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t usedCoreNums;
    int64_t tileA0Outer;
    int64_t tileA0Len;
    int64_t tileA0Tail;
    int64_t binAddRFactor;
    int64_t binAddRLoop;
    int64_t binAddRTotalLoop;
    int64_t binAddRTail;
    int64_t binAddBasicBlockLoop;
    int64_t binAddMainFoldCount;
    int64_t binAddCacheBufferCount;
    int64_t binAddResultCacheID;
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