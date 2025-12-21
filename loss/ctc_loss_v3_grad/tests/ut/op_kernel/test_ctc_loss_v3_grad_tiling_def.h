/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RMS_NORM_TILING_H_
#define _RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct CTCLossV3GradTilingDataTest {
    int64_t sliceLength;
    int64_t sliceLengthTail;
    int64_t probSliceNum;
    int64_t alphaLength;
    int64_t maxInputLength;
    int64_t symbolSet;
    int64_t batchSize;
    int64_t sDimRange;
    int64_t targetsDimNum;
    int64_t targetsNum;
    int64_t taskPerCore;
    int64_t taskTailCore;
    int64_t BLANK;
    int64_t zeroInfinity;
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                               \
    CTCLossV3GradTilingDataTest tilingData;                                          \
    INIT_TILING_DATA(CTCLossV3GradTilingDataTest, tilingDataPointer, tilingPointer); \
    (tilingData).sliceLength = tilingDataPointer->sliceLength;                   \
    (tilingData).sliceLengthTail = tilingDataPointer->sliceLengthTail;           \
    (tilingData).probSliceNum = tilingDataPointer->probSliceNum;                 \
    (tilingData).alphaLength = tilingDataPointer->alphaLength;                   \
    (tilingData).maxInputLength = tilingDataPointer->maxInputLength;             \
    (tilingData).symbolSet = tilingDataPointer->symbolSet;                       \
    (tilingData).batchSize = tilingDataPointer->batchSize;                       \
    (tilingData).sDimRange = tilingDataPointer->sDimRange;                       \
    (tilingData).targetsDimNum = tilingDataPointer->targetsDimNum;               \
    (tilingData).targetsNum = tilingDataPointer->targetsNum;                     \
    (tilingData).taskPerCore = tilingDataPointer->taskPerCore;                   \
    (tilingData).taskTailCore = tilingDataPointer->taskTailCore;                 \
    (tilingData).BLANK = tilingDataPointer->BLANK;                               \
    (tilingData).zeroInfinity = tilingDataPointer->zeroInfinity;

#define DTYPE_GRAD float
#define DTYPE_TARGETS int64_t

#endif