/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _APPLY_ADAM_W_V2_TILING_H_
#define _APPLY_ADAM_W_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct ApplyAdamWV2TilingData {
    int64_t totalCoreNum;
    int64_t handleExtraLoopCoreNum;
    int64_t usedCoreNum;
    int64_t numPerLoop;
    int64_t loopNumPerCore;
    int64_t numLastLoop;
    int64_t isBfloat16;
    float lr;
    float beta1;
    float beta2;
    float weightDecay;
    float eps;
    int64_t amsgrad;
    int64_t maximize;
    int64_t tilingKey;
};

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    ApplyAdamWV2TilingData tilingData;                                               \
    INIT_TILING_DATA(ApplyAdamWV2TilingData, tilingDataPointer, tilingPointer);      \
    (tilingData).totalCoreNum = tilingDataPointer->totalCoreNum;                     \
    (tilingData).handleExtraLoopCoreNum = tilingDataPointer->handleExtraLoopCoreNum; \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                       \
    (tilingData).numPerLoop = tilingDataPointer->numPerLoop;                         \
    (tilingData).loopNumPerCore = tilingDataPointer->loopNumPerCore;                 \
    (tilingData).numLastLoop = tilingDataPointer->numLastLoop;                       \
    (tilingData).isBfloat16 = tilingDataPointer->isBfloat16;                         \
    (tilingData).lr = tilingDataPointer->lr;                                         \
    (tilingData).beta1 = tilingDataPointer->beta1;                                   \
    (tilingData).beta2 = tilingDataPointer->beta2;                                   \
    (tilingData).weightDecay = tilingDataPointer->weightDecay;                       \
    (tilingData).eps = tilingDataPointer->eps;                                       \
    (tilingData).amsgrad = tilingDataPointer->amsgrad;                               \
    (tilingData).maximize = tilingDataPointer->maximize;                             \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;

#endif // _APPLY_ADAM_W_V2_TILING_H_
