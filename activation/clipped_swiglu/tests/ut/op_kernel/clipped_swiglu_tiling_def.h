/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _CLIPPED_SWIGLU_TILING_H_
#define _CLIPPED_SWIGLU_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_X DT_BF16
#define __CCE_UT_TEST__
#define __aicore__

#pragma pack(1)
struct ClippedSwigluTilingData {
    int64_t coreNumAll;
    int64_t dimBatchSize;
    int64_t dim2H;
    int64_t isLongH;
    int64_t isGroup;
    int64_t isInterleaved;
    float   gluAlpha;
    float   gluLimit;
    float   gluBias;
    int64_t ubMaxPair;
    int64_t groupNum;
};

#pragma pack()

inline void IClippedSwigluTilingData(uint8_t* tiling, ClippedSwigluTilingData* constData)
{
    memcpy(constData, tiling, sizeof(ClippedSwigluTilingData));
}



#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg)      \
    tilingStruct tilingData;                                                  \
    IClippedSwigluTilingData(tilingArg, &tilingData)

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
    __ubuf__ tilingStruct* tilingDataPointer =                                 \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    ClippedSwigluTilingData tilingData;                                               \
    INIT_TILING_DATA(ClippedSwigluTilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).coreNumAll = tilingDataPointer->coreNumAll;                    \
    (tilingData).dimBatchSize = tilingDataPointer->dimBatchSize;                  \
    (tilingData).dim2H = tilingDataPointer->dim2H;   \
    (tilingData).isLongH = tilingDataPointer->isLongH;         \
    (tilingData).isGroup = tilingDataPointer->isGroup;   \
    (tilingData).isInterleaved = tilingDataPointer->isInterleaved; \
    (tilingData).gluAlpha = tilingDataPointer->gluAlpha;      \
    (tilingData).gluLimit = tilingDataPointer->gluLimit;      \
    (tilingData).gluBias = tilingDataPointer->gluBias;  \
    (tilingData).ubMaxPair = tilingDataPointer->ubMaxPair;  \
    (tilingData).groupNum = tilingDataPointer->groupNum;

// #define DTYPE_X DT_BF16

#endif