/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _SCALED_MASKED_SOFTMAX_GRAD_V2_TILING_H_
#define _SCALED_MASKED_SOFTMAX_GRAD_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define __CCE_UT_TEST__

#pragma pack(1)

struct ScaledMaskedSoftmaxGradV2TilingData {
    uint64_t usedCoreNum;
    uint64_t batch;
    uint64_t channel;
    uint64_t seqLength;
    uint64_t headDim;
    uint64_t totalLine;
    uint64_t paddedHeadDim;
    uint64_t totalLinePerHeadCore;
    uint64_t totalLinePerTailCore;
    uint64_t maxLinePerLoop;
    uint64_t tailLinePerHeadCore;
    uint64_t tailLinePerTailCore;
    uint64_t headCoreNum;
    uint64_t maskMoveMode;
    uint64_t selectSize;

    float scale;
    uint32_t fixedTriuMask;
    SoftMaxTiling softmaxGradTilingData;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)               \
        __ubuf__ tilingStruct* tilingDataPointer =                                        \
            reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                               \
        ScaledMaskedSoftmaxGradV2TilingData tilingData;                                          \
        INIT_TILING_DATA(ScaledMaskedSoftmaxGradV2TilingData, tilingDataPointer, tilingPointer); \
        (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                               \
        (tilingData).batch = tilingDataPointer->batch;                                           \
        (tilingData).channel = tilingDataPointer->channel;                                       \
        (tilingData).seqLength = tilingDataPointer->seqLength;                                   \
        (tilingData).headDim = tilingDataPointer->headDim;                                       \
        (tilingData).totalLine = tilingDataPointer->totalLine;                                   \
        (tilingData).paddedHeadDim = tilingDataPointer->paddedHeadDim;                           \
        (tilingData).totalLinePerHeadCore = tilingDataPointer->totalLinePerHeadCore;             \
        (tilingData).totalLinePerTailCore = tilingDataPointer->totalLinePerTailCore;             \
        (tilingData).maxLinePerLoop = tilingDataPointer->maxLinePerLoop;                         \
        (tilingData).tailLinePerHeadCore = tilingDataPointer->tailLinePerHeadCore;               \
        (tilingData).tailLinePerTailCore = tilingDataPointer->tailLinePerTailCore;               \
        (tilingData).headCoreNum = tilingDataPointer->headCoreNum;                               \
        (tilingData).maskMoveMode = tilingDataPointer->maskMoveMode;                             \
        (tilingData).selectSize = tilingDataPointer->selectSize;                                 \
        (tilingData).scale = tilingDataPointer->scale;                                           \
        (tilingData).fixedTriuMask = tilingDataPointer->fixedTriuMask;                           \
        (tilingData).softmaxGradTilingData = tilingDataPointer->softmaxGradTilingData;
#endif