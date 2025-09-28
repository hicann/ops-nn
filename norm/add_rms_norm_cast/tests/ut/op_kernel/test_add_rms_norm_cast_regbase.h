/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_CAST_REGBASE_TILING_H_
#define ADD_RMS_NORM_CAST_REGBASE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define DTYPE_X1 bfloat16_t

#pragma pack(1)

struct AddRmsNormCastRegbaseTilingData {
    uint64_t numM = 0;
    uint64_t numN = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseNDtypeAlign = 0;
    uint64_t baseNReduceAlign = 0;
    uint64_t powerSplit = 0;
    uint64_t powerLoop = 0;
    uint64_t mPerCore = 0;
    uint64_t mLastCore = 0;
    float epsilon = 0;
    float avgFactor = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    AddRmsNormCastRegbaseTilingData tilingData;                                          \
    INIT_TILING_DATA(AddRmsNormCastRegbaseTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numM = tilingDataPointer->numM;                                         \
    (tilingData).numN = tilingDataPointer->numN;                                         \
    (tilingData).baseM = tilingDataPointer->baseM;                                       \
    (tilingData).baseN = tilingDataPointer->baseN;                                       \
    (tilingData).baseNDtypeAlign = tilingDataPointer->baseNDtypeAlign;                   \
    (tilingData).baseNReduceAlign = tilingDataPointer->baseNReduceAlign;                 \
    (tilingData).powerSplit = tilingDataPointer->powerSplit;                             \
    (tilingData).powerLoop = tilingDataPointer->powerLoop;                               \
    (tilingData).mPerCore = tilingDataPointer->mPerCore;                                 \
    (tilingData).mLastCore = tilingDataPointer->mLastCore;                               \
    (tilingData).epsilon = tilingDataPointer->epsilon;                                   \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AddRmsNormCastRegbaseTilingData* constData)
{
    const __gm__ int64_t* src = (const __gm__ int64_t*)tiling;
    int64_t* dst = (int64_t*)constData;
    for (auto i = 0; i < sizeof(AddRmsNormCastRegbaseTilingData) / sizeof(int64_t); i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AddRmsNormCastRegbaseTilingData* constData)
{
    memcpy(constData, tiling, sizeof(AddRmsNormCastRegbaseTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define DTYPE_X1 half

#endif
