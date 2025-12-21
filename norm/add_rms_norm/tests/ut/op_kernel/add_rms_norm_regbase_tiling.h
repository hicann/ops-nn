/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_REGBASE_TILING_H_
#define ADD_RMS_NORM_REGBASE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormRegbaseTilingData {
    uint32_t numRow = 0;
    uint32_t numCol = 0;
    uint32_t numColAlign = 0;
    uint32_t blockFactor = 0;
    uint32_t rowFactor = 0;
    uint32_t ubFactor = 0;
    uint32_t ubLoop = 0;
    uint32_t colBuferLength = 0;
    uint32_t multiNNum = 0;
    float epsilon = 0;
    float avgFactor = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    AddRMSNormRegbaseTilingData tilingData;                                          \
    INIT_TILING_DATA(AddRMSNormRegbaseTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numRow = tilingDataPointer->numRow;                                 \
    (tilingData).numCol = tilingDataPointer->numCol;                                 \
    (tilingData).numColAlign = tilingDataPointer->numColAlign;                       \
    (tilingData).blockFactor = tilingDataPointer->blockFactor;                       \
    (tilingData).rowFactor = tilingDataPointer->rowFactor;                           \
    (tilingData).ubFactor = tilingDataPointer->ubFactor;                             \
    (tilingData).ubLoop = tilingDataPointer->ubLoop;                                 \
    (tilingData).colBuferLength = tilingDataPointer->colBuferLength;                 \
    (tilingData).multiNNum = tilingDataPointer->multiNNum;                           \
    (tilingData).epsilon = tilingDataPointer->epsilon;                               \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AddRMSNormRegbaseTilingData* constData)
{
    const __gm__ int64_t* src = (const __gm__ int64_t*)tiling;
    int64_t* dst = (int64_t*)constData;
    for (auto i = 0; i < sizeof(AddRMSNormRegbaseTilingData) / sizeof(int64_t); i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AddRMSNormRegbaseTilingData* constData)
{
    memcpy(constData, tiling, sizeof(AddRMSNormRegbaseTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif
