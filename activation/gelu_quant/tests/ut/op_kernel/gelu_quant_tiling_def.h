/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _GELU_QUANT_TILING_DEF_H_
#define _GELU_QUANT_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_UT_TEST__

#pragma pack(1)

struct GeluQuantTilingData {
    uint32_t usedCoreNum;
    uint32_t normalCoreProcessNum;
    uint32_t tailCoreProcessNum;
    uint32_t endAxisLen;
    uint32_t endAxisLenAligned;
    uint32_t coexistentNodeNum;
    uint32_t coexistentNodeElementNum;
    uint32_t rowInner;
    uint32_t rowTail;
    uint32_t rowOuter;
    uint32_t colInner;
    uint32_t colTail;
    uint32_t colOuter;
    uint32_t quantMode;
    uint32_t approximate;
    uint32_t inputScaleType;
    uint32_t inputOffsetType;
    uint32_t tilingKey;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    GeluQuantTilingData tilingData;                                                      \
    INIT_TILING_DATA(GeluQuantTilingData, tilingDataPointer, tilingPointer);             \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                           \
    (tilingData).normalCoreProcessNum = tilingDataPointer->normalCoreProcessNum;         \
    (tilingData).tailCoreProcessNum = tilingDataPointer->tailCoreProcessNum;             \
    (tilingData).endAxisLen = tilingDataPointer->endAxisLen;                             \
    (tilingData).endAxisLenAligned = tilingDataPointer->endAxisLenAligned;               \
    (tilingData).coexistentNodeNum = tilingDataPointer->coexistentNodeNum;               \
    (tilingData).coexistentNodeElementNum = tilingDataPointer->coexistentNodeElementNum; \
    (tilingData).rowInner = tilingDataPointer->rowInner;                                 \
    (tilingData).rowTail = tilingDataPointer->rowTail;                                   \
    (tilingData).rowOuter = tilingDataPointer->rowOuter;                                 \
    (tilingData).colInner = tilingDataPointer->colInner;                                 \
    (tilingData).colTail = tilingDataPointer->colTail;                                   \
    (tilingData).colOuter = tilingDataPointer->colOuter;                                 \
    (tilingData).quantMode = tilingDataPointer->quantMode;                               \
    (tilingData).approximate = tilingDataPointer->approximate;                           \
    (tilingData).inputScaleType = tilingDataPointer->inputScaleType;                     \
    (tilingData).inputOffsetType = tilingDataPointer->inputOffsetType;                   \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;

#endif