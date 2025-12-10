/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef _TEST_ASCEND_QUANT_V2_H_
#define _TEST_ASCEND_QUANT_V2_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct AscendQuantV2TilingData {
    uint32_t numCore;
    int32_t blockAxis;
    int32_t ubAxis;
    int64_t dim0;
    int64_t dim1;
    int64_t dim2;
    int64_t blockUnion;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseN;
    int64_t baseLen;
    int16_t hasOffset;
    int16_t sqrtMode;
    int16_t roundMode;
    int16_t reserve1;
    int64_t E;
    int64_t K;
    int64_t N;
    int64_t needCoreNum;
};

#pragma pack()

#define DTYPE_X int64_t
#define ORIG_DTYPE_Y 29

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                               \
    AscendQuantV2TilingData tilingData;                                          \
    INIT_TILING_DATA(AscendQuantV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                           \
    (tilingData).blockAxis = tilingDataPointer->blockAxis;                       \
    (tilingData).ubAxis = tilingDataPointer->ubAxis;                             \
    (tilingData).dim0 = tilingDataPointer->dim0;                                 \
    (tilingData).dim1 = tilingDataPointer->dim1;                                 \
    (tilingData).dim2 = tilingDataPointer->dim2;                                 \
    (tilingData).blockUnion = tilingDataPointer->blockUnion;                     \
    (tilingData).blockFactor = tilingDataPointer->blockFactor;                   \
    (tilingData).blockTailFactor = tilingDataPointer->blockTailFactor;           \
    (tilingData).baseN = tilingDataPointer->baseN;                               \
    (tilingData).baseLen = tilingDataPointer->baseLen;                           \
    (tilingData).hasOffset = tilingDataPointer->hasOffset;                       \
    (tilingData).sqrtMode = tilingDataPointer->sqrtMode;                         \
    (tilingData).roundMode = tilingDataPointer->roundMode;                       \
    (tilingData).reserve1 = tilingDataPointer->reserve1;                         \
    (tilingData).E = tilingDataPointer->E;                                       \
    (tilingData).K = tilingDataPointer->K;                                       \
    (tilingData).N = tilingDataPointer->N;                                       \
    (tilingData).needCoreNum = tilingDataPointer->needCoreNum;
#endif