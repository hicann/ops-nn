/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef DUA_QUANTIZE_ADD_LAYER_NORM_NORM_TILING_H_
#define DUA_QUANTIZE_ADD_LAYER_NORM_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct DuaQuantizeAddLayerNormTilingData {
    uint32_t numCore = 40;
    uint32_t numLastDim = 11264;
    uint32_t numRow = 1024;
    uint32_t nlFirstDimPerCore = 26;
    uint32_t lFirstDimPerCore = 10;
    uint32_t firstDimPerTime = 1;
    uint32_t lastDimPerTime = 1;
    float aveNum = 0.00008878;
    float eps = 0.00001;
    uint32_t colMoveCnt = 1;
    uint32_t colTail = 1;
    uint32_t isZeroPoint1Exist = 1;
    uint32_t isZeroPoint2Exist = 1;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                         \
    DuaQuantizeAddLayerNormTilingData tilingData;                                          \
    INIT_TILING_DATA(DuaQuantizeAddLayerNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                                     \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                               \
    (tilingData).numRow = tilingDataPointer->numRow;                                       \
    (tilingData).nlFirstDimPerCore = tilingDataPointer->nlFirstDimPerCore;                 \
    (tilingData).lFirstDimPerCore = tilingDataPointer->lFirstDimPerCore;                   \
    (tilingData).firstDimPerTime = tilingDataPointer->firstDimPerTime;                     \
    (tilingData).lastDimPerTime = tilingDataPointer->lastDimPerTime;                       \
    (tilingData).aveNum = tilingDataPointer->aveNum;                                       \
    (tilingData).eps = tilingDataPointer->eps;                                             \
    (tilingData).colMoveCnt = tilingDataPointer->colMoveCnt;                               \
    (tilingData).colTail = tilingDataPointer->colTail;                                     \
    (tilingData).isZeroPoint1Exist = tilingDataPointer->isZeroPoint1Exist;                 \
    (tilingData).isZeroPoint2Exist = tilingDataPointer->isZeroPoint2Exist;
#endif