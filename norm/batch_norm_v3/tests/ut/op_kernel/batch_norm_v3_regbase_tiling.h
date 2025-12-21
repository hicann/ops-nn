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
 * \file test_batch_norm_v3.h
 * \brief
 */

#ifndef BATCH_NORM_V3_REGBASE_TILING_H_
#define BATCH_NORM_V3_REGBASE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)
struct BatchNormV3WelfordRegbaseTilingData {
    int64_t r1 = 0;
    int64_t r0 = 0;
    int64_t a0 = 0;
    int64_t loopR1outer = 0;
    int64_t r1Factor = 0;
    int64_t loopR0outer = 0;
    int64_t r0Factor = 0;
    int64_t realCoreNum = 0;
    int64_t numLastCore = 0;
    int64_t aBlockFactor = 0;
    int64_t aGatherLimit = 0;
    int64_t parallelN = 0;
    int64_t processSize = 0;
    int64_t ubSize = 0;
    int64_t elemNum = 0;
    int64_t vlLenFp32 = 0;
    int64_t cutR1OrR0 = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLastNum = 0;
    int64_t binaryAddQuotient = 0;
    float epsilon = 0;
    float momentum = 0;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3WelfordRegbaseTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3WelfordRegbaseTilingData));
}

#pragma pack(1)
struct BatchNormV3RAFullReduceTilingData {
    int64_t r1 = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0;
    float momentum = 0;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3RAFullReduceTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3RAFullReduceTilingData));
}

#pragma pack(1)
struct BatchNormV3FullReduceRegbaseTilingData {
    int64_t r1 = 0;
    int64_t r0 = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t r1r0LoopCount = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLastNum = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0;
    float momentum = 0;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3FullReduceRegbaseTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3FullReduceRegbaseTilingData));
}

#pragma pack(1)
struct BatchNormV3RAWelfordTilingData {
    int64_t r = 0;
    int64_t rFactor = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLast = 0;
    int64_t powerOfTwoForR = 0;
    float epsilon = 0;
    float momentum = 0;
};
#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3RAWelfordTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3RAWelfordTilingData));
}

#pragma pack(1)
struct BatchNormV3InferLastChannelTilingData {
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
#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3InferLastChannelTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3InferLastChannelTilingData));
}

#pragma pack(1)
struct BatchNormV3InferTilingData {
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
#pragma pack()

inline void InitTilingData(uint8_t* tiling, BatchNormV3InferTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchNormV3InferTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif