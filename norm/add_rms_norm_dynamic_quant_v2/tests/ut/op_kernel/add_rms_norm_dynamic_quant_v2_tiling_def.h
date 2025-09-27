/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_V2_TILING_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define DTYPE_X1 bfloat16_t

#pragma pack(1)

struct AddRmsNormDynamicQuantV2TilingData {
    uint64_t useCore = 0;
    uint64_t numFirstDim = 0;
    uint64_t numLastDim = 0;
    uint64_t numLastDimAligned = 0;
    uint64_t firstDimPerCore = 0;
    uint64_t firstDimPerCoreTail = 0;
    uint64_t firstDimPerLoop = 0;
    uint64_t lastDimLoopNum = 0;
    uint64_t lastDimSliceLen = 0;
    uint64_t lastDimSliceLenTail = 0;
    uint32_t smoothNum = 0;
    float epsilon = 0;
    float avgFactor = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                          \
    AddRmsNormDynamicQuantV2TilingData tilingData;                                          \
    INIT_TILING_DATA(AddRmsNormDynamicQuantV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).useCore = tilingDataPointer->useCore;                                      \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                              \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                                \
    (tilingData).numLastDimAligned = tilingDataPointer->numLastDimAligned;                  \
    (tilingData).firstDimPerCore = tilingDataPointer->firstDimPerCore;                      \
    (tilingData).firstDimPerCoreTail = tilingDataPointer->firstDimPerCoreTail;              \
    (tilingData).firstDimPerLoop = tilingDataPointer->firstDimPerLoop;                      \
    (tilingData).lastDimLoopNum = tilingDataPointer->lastDimLoopNum;                        \
    (tilingData).lastDimSliceLen = tilingDataPointer->lastDimSliceLen;                      \
    (tilingData).lastDimSliceLenTail = tilingDataPointer->lastDimSliceLenTail;              \
    (tilingData).smoothNum = tilingDataPointer->smoothNum;                                  \
    (tilingData).epsilon = tilingDataPointer->epsilon;                                      \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;
#endif