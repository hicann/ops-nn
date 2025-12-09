/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef QUANTIZE_ADD_LAYER_NORM_NORM_TILING_H_
#define QUANTIZE_ADD_LAYER_NORM_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct QuantizeAddLayerNormTilingData {
    uint32_t numCore = 40;
    uint32_t numLastDim = 11264;
    uint32_t numFirstDim = 1024;
    uint32_t firstDimPerCore = 26;
    uint32_t firstDimPerCoreTail = 10;
    uint32_t firstDimPerTime = 1;
    float aveFactor = 0.00008878;
    float eps = 0.00001;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                      \
    QuantizeAddLayerNormTilingData tilingData;                                          \
    INIT_TILING_DATA(QuantizeAddLayerNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                                  \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                            \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                          \
    (tilingData).firstDimPerCore = tilingDataPointer->firstDimPerCore;                  \
    (tilingData).firstDimPerCoreTail = tilingDataPointer->firstDimPerCoreTail;          \
    (tilingData).aveFactor = tilingDataPointer->aveFactor;                              \
    (tilingData).eps = tilingDataPointer->eps;
#endif