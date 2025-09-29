/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RMS_NORM_TILING_H_
#define _RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct CTCLossV3TilingDataTest {
    int64_t sliceLength;
    int64_t sliceLengthTail;
    int64_t probSliceNum;
    int64_t maxTargetLength;
    int64_t symbleSet;
    int64_t batchSize;
    int64_t timeStep;
    int64_t targetsDimNum;
    int64_t targetsDimLength;
    int64_t targetsNum;
    int64_t taskPerCore;
    int64_t taskTailCore;
    int64_t blank;
    int64_t zeroInfinity;
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
    CTCLossV3TilingDataTest tilingData;                                          \
    INIT_TILING_DATA(CTCLossV3TilingDataTest, tilingDataPointer, tilingPointer); \
    (tilingData).sliceLength = tilingDataPointer->sliceLength;               \
    (tilingData).sliceLengthTail = tilingDataPointer->sliceLengthTail;       \
    (tilingData).probSliceNum = tilingDataPointer->probSliceNum;             \
    (tilingData).maxTargetLength = tilingDataPointer->maxTargetLength;       \
    (tilingData).symbleSet = tilingDataPointer->symbleSet;                   \
    (tilingData).batchSize = tilingDataPointer->batchSize;                   \
    (tilingData).timeStep = tilingDataPointer->timeStep;                     \
    (tilingData).targetsDimNum = tilingDataPointer->targetsDimNum;           \
    (tilingData).targetsDimLength = tilingDataPointer->targetsDimLength;     \
    (tilingData).targetsNum = tilingDataPointer->targetsNum;                 \
    (tilingData).taskPerCore = tilingDataPointer->taskPerCore;               \
    (tilingData).taskTailCore = tilingDataPointer->taskTailCore;             \
    (tilingData).blank = tilingDataPointer->blank;                           \
    (tilingData).zeroInfinity = tilingDataPointer->zeroInfinity;

#define DTYPE_LOG_PROBS float
#define DTYPE_TARGETS int64_t

#endif