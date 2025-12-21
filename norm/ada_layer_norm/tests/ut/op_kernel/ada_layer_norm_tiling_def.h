/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_tiling.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_TILING_DEFH
#define ADA_LAYER_NORM_TILING_DEFH

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#define DTYPE_X bfloat16_t

struct AdaLayerNormTilingData {
    int64_t batchSize;
    int64_t seqLen;
    int64_t hiddenDim;
    float epsilon;
    float aveFactor;
    int32_t hasWeight;
    int32_t hasBias;
    int32_t hasSmooth;
    int64_t singleCoreNum;
    int64_t tailNum;
    int64_t sliceSize;
    int64_t rowNum;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                              \
    AdaLayerNormTilingData tilingData;                                          \
    INIT_TILING_DATA(AdaLayerNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).batchSize = tilingDataPointer->batchSize;                      \
    (tilingData).seqLen = tilingDataPointer->seqLen;                            \
    (tilingData).hiddenDim = tilingDataPointer->hiddenDim;                      \
    (tilingData).epsilon = tilingDataPointer->epsilon;                          \
    (tilingData).aveFactor = tilingDataPointer->aveFactor;                      \
    (tilingData).hasWeight = tilingDataPointer->hasWeight;                      \
    (tilingData).hasBias = tilingDataPointer->hasBias;                          \
    (tilingData).hasSmooth = tilingDataPointer->hasSmooth;                      \
    (tilingData).singleCoreNum = tilingDataPointer->singleCoreNum;              \
    (tilingData).tailNum = tilingDataPointer->tailNum;                          \
    (tilingData).sliceSize = tilingDataPointer->sliceSize;                      \
    (tilingData).rowNum = tilingDataPointer->rowNum;

#endif // ADA_LAYER_NORM_TILING_DEFH