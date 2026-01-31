/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_QUANT_V2_TILING_H_
#define RMS_NORM_QUANT_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include <climits>

// #define DT_BF16 bfloat16_t
// #define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RmsNormQuantV2TilingData {
    uint32_t reverse;
};

struct RmsNormQuantV2RegbaseFullLoadTilingData {
    int64_t a;           // a
    int64_t r;           // r
    int64_t q;           // q 传入的scales值，1或者R
    int64_t blockFactor; //单核处理a行数
    int64_t blockTail;   //尾核处理a行数
    int64_t ubFactor;    // ub处理a行数
    int64_t binaryAdd;   // r轴二分累加折叠点
    uint64_t optionMask; // scales2  zero_points1  zero_points2 beta是否存在
    int64_t divMode;     //量化参数
    int64_t dstDtype;    //输出类型
    float epsilon;
    float avgFactor; // avg_value  1/R
};

struct RmsNormQuantV2RegbaseRecomputeTilingData {
    int64_t numM;
    int64_t numN;
    int64_t baseM;
    int64_t baseN;
    int64_t mPerCore;       // 单核处理 A 行数
    int64_t mLastCore;      // 尾核处理 A 行数
    int64_t nUbLoops;       // ub 处理 r 轴循环次数
    int64_t binAddQuotient; // ub整块二分累加折叠点
    int64_t powerSplit;     // R 轴 二分点
    int64_t mainFoldCount;  // 折叠部分的主块长度
    int64_t foldTail;       // 折叠部分的尾块
    uint64_t optionMask;    // 5个可选参数. needBrc | scales2 | zero_points1 | zero_points2 | beta
    uint64_t divMode;       // 量化模式
    uint64_t dstDtype;      // 输出类型
    float epsilon;
    float avgFactor; // 1 / R
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                \
    RmsNormQuantV2TilingData tilingData;                                          \
    INIT_TILING_DATA(RmsNormQuantV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).reverse = tilingDataPointer->reverse;

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#define DTYPE_X float
#define DTYPE_SCALES1 float
#define DTYPE_ZERO_POINTS1 float
#define DTYPE_ZERO_POINTS2 float
#define DTYPE_Y1 int8_t

#endif
