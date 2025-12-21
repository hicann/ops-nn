/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_NORM_REGBASE_TILING_H_
#define ADD_LAYER_NORM_NORM_REGBASE_TILING_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddLayerNormRegbaseTilingData {
    uint32_t blockSize;
    uint32_t usedCoreNum;
    uint32_t vlFp32;
    uint32_t tailCoreStartIndex;
    int64_t rowsPerCore;
    int64_t rowsPerTailCore;
    int64_t rowsPerLoop;
    int64_t cols;
    int64_t colsPerLoop;
    int64_t colsLoopCount;
    int64_t colsTail;
    int64_t binaryAddNum;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;
    float eps;
    uint32_t outputX;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                     \
    AddLayerNormRegbaseTilingData tilingData;                                          \
    INIT_TILING_DATA(AddLayerNormRegbaseTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).blockSize = tilingDataPointer->blockSize;                             \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                         \
    (tilingData).vlFp32 = tilingDataPointer->vlFp32;                                   \
    (tilingData).tailCoreStartIndex = tilingDataPointer->tailCoreStartIndex;           \
    (tilingData).rowsPerCore = tilingDataPointer->rowsPerCore;                         \
    (tilingData).rowsPerTailCore = tilingDataPointer->rowsPerTailCore;                 \
    (tilingData).rowsPerLoop = tilingDataPointer->rowsPerLoop;                         \
    (tilingData).cols = tilingDataPointer->cols;                                       \
    (tilingData).colsPerLoop = tilingDataPointer->colsPerLoop;                         \
    (tilingData).colsLoopCount = tilingDataPointer->colsLoopCount;                     \
    (tilingData).colsTail = tilingDataPointer->colsTail;                               \
    (tilingData).binaryAddNum = tilingDataPointer->binaryAddNum;                       \
    (tilingData).binaryAddK = tilingDataPointer->binaryAddK;                           \
    (tilingData).binaryAddLastNum = tilingDataPointer->binaryAddLastNum;               \
    (tilingData).eps = tilingDataPointer->eps;                                         \
    (tilingData).outputX = tilingDataPointer->outputX;

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AddLayerNormRegbaseTilingData* constData)
{
    const __gm__ int64_t* src = (const __gm__ int64_t*)tiling;
    int64_t* dst = (int64_t*)constData;
    for (auto i = 0; i < sizeof(AddLayerNormRegbaseTilingData) / sizeof(int64_t); i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AddLayerNormRegbaseTilingData* constData)
{
    memcpy(constData, tiling, sizeof(AddLayerNormRegbaseTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    AddLayerNormRegbaseTilingData tilingData;  \
    InitTilingData(tilingArg, &tilingData)

#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_GAMMA float
#define DTYPE_BETA float
#define DTYPE_BIAS float
#define DTYPE_Y float
#define DTYPE_X float

#endif // _ADD_LAYER_NORM_NORM_REGBASE_TILING_H_
