/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FAKE_QUANT_WITH_MIN_MAX_ARGS_TILING_DEF_H_
#define _FAKE_QUANT_WITH_MIN_MAX_ARGS_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_UT_TEST__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        (__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                         \
    FakeQuantWithMinMaxArgsTilingData tilingData;                                          \
    INIT_TILING_DATA(FakeQuantWithMinMaxArgsTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).totalLen = tilingDataPointer->totalLen;                                   \
    (tilingData).numCore = tilingDataPointer->numCore;                                     \
    (tilingData).blockFactor = tilingDataPointer->blockFactor;                             \
    (tilingData).blockTailFactor = tilingDataPointer->blockTailFactor;                     \
    (tilingData).baseLen = tilingDataPointer->baseLen;                                     \
    (tilingData).nudgedMin = tilingDataPointer->nudgedMin;                                 \
    (tilingData).nudgedMax = tilingDataPointer->nudgedMax;                                 \
    (tilingData).scale = tilingDataPointer->scale;                                         \
    (tilingData).scaleInv = tilingDataPointer->scaleInv;                                   \
    (tilingData).quantZero = tilingDataPointer->quantZero;

#endif