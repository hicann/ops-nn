/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GE_SCALED_MASKED_SOFTMAX_V2_TILING_H_
#define _GE_SCALED_MASKED_SOFTMAX_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)

struct ScaledMaskedSoftmaxV2TilingData {
    uint64_t coreNum;
    uint64_t batch;
    uint64_t channel;
    uint64_t height;
    uint64_t width;

    uint64_t maskBatch;
    uint64_t maskChannel;
    uint64_t maskHeight;
    uint64_t maskWidth;

    float scale;
    uint64_t maskMode;
    uint64_t paddingNum;
    uint64_t padLineNum;
    uint64_t alignedMaskPadding;
    uint64_t alignedMaskWidth;

    uint64_t nStep;
    uint64_t cStep;

    uint64_t headCoreNum;

    uint64_t lineHeadCore;
    uint64_t iterHeadCore;
    uint64_t lineHeadIter;
    uint64_t lineLastHeadIter;

    uint64_t lineTailCore;
    uint64_t iterTailCore;
    uint64_t lineTailIter;
    uint64_t lineLastTailIter;

    SoftMaxTiling softmaxTilingData;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    ScaledMaskedSoftmaxV2TilingData tilingData;                                          \
    INIT_TILING_DATA(ScaledMaskedSoftmaxV2TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreNum = tilingDataPointer->coreNum;                                   \
    (tilingData).batch = tilingDataPointer->batch;                                       \
    (tilingData).channel = tilingDataPointer->channel;                                   \
    (tilingData).height = tilingDataPointer->height;                                     \
    (tilingData).width = tilingDataPointer->width;                                       \
    (tilingData).maskBatch = tilingDataPointer->maskBatch;                               \
    (tilingData).maskChannel = tilingDataPointer->maskChannel;                           \
    (tilingData).maskHeight = tilingDataPointer->maskHeight;                             \
    (tilingData).maskWidth = tilingDataPointer->maskWidth;                               \
    (tilingData).scale = tilingDataPointer->scale;                                       \
    (tilingData).maskMode = tilingDataPointer->maskMode;                                 \
    (tilingData).paddingNum = tilingDataPointer->paddingNum;                             \
    (tilingData).padLineNum = tilingDataPointer->padLineNum;                             \
    (tilingData).alignedMaskPadding = tilingDataPointer->alignedMaskPadding;             \
    (tilingData).alignedMaskWidth = tilingDataPointer->alignedMaskWidth;                 \
    (tilingData).nStep = tilingDataPointer->nStep;                                       \
    (tilingData).cStep = tilingDataPointer->cStep;                                       \
    (tilingData).headCoreNum = tilingDataPointer->headCoreNum;                           \
    (tilingData).lineHeadCore = tilingDataPointer->lineHeadCore;                         \
    (tilingData).iterHeadCore = tilingDataPointer->iterHeadCore;                         \
    (tilingData).lineHeadIter = tilingDataPointer->lineHeadIter;                         \
    (tilingData).lineLastHeadIter = tilingDataPointer->lineLastHeadIter;                 \
    (tilingData).lineTailCore = tilingDataPointer->lineTailCore;                         \
    (tilingData).iterTailCore = tilingDataPointer->iterTailCore;                         \
    (tilingData).lineTailIter = tilingDataPointer->lineTailIter;                         \
    (tilingData).lineLastTailIter = tilingDataPointer->lineLastTailIter;                 \
    (tilingData).softmaxTilingData = tilingDataPointer->softmaxTilingData;
#endif
