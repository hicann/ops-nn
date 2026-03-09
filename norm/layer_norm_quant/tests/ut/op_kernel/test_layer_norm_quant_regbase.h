/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAYER_NORM_NORM_QUANT_REGBASE_TILING_H_
#define LAYER_NORM_NORM_QUANT_REGBASE_TILING_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct LayerNormQuantRegTilingData {
    uint32_t numCore;
    uint32_t numLastDim;
    uint32_t numFirstDim;
    uint32_t nlFirstdimPerCore;
    uint32_t lFirstdimPerCore;
    uint32_t firstDimPerTimes;
    uint32_t colsAligned;
    float epsStr;
    float aveStr;
    uint32_t sliceNum;
    uint32_t sliceSize;
    uint32_t tailSliceSize;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                          \
    LayerNormQuantRegTilingData tilingData;                                          \
    INIT_TILING_DATA(LayerNormQuantRegTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                                  \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                              \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                                        \
    (tilingData).nlFirstdimPerCore = tilingDataPointer->nlFirstdimPerCore;                              \
    (tilingData).lFirstdimPerCore = tilingDataPointer->lFirstdimPerCore;                      \
    (tilingData).firstDimPerTimes = tilingDataPointer->firstDimPerTimes;                              \
    (tilingData).colsAligned = tilingDataPointer->colsAligned;                                            \
    (tilingData).epsStr = tilingDataPointer->epsStr;                              \
    (tilingData).aveStr = tilingDataPointer->aveStr;                          \
    (tilingData).sliceNum = tilingDataPointer->sliceNum;                                    \
    (tilingData).sliceSize = tilingDataPointer->sliceSize;                            \
    (tilingData).tailSliceSize = tilingDataPointer->tailSliceSize;

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormQuantRegTilingData* constData)
{
    const __gm__ int64_t* src = (const __gm__ int64_t*)tiling;
    int64_t* dst = (int64_t*)constData;
    for (auto i = 0; i < sizeof(LayerNormQuantRegTilingData) / sizeof(int64_t); i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormQuantRegTilingData* constData)
{
    memcpy(constData, tiling, sizeof(LayerNormQuantRegTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg)     \
    LayerNormQuantRegTilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)

#define DTYPE_X float
#define DTYPE_GAMMA float
#define DTYPE_BETA float
#define DTYPE_Y float
#define DTYPE_SCALES float

#endif // LAYER_NORM_NORM_QUANT_REGBASE_TILING_H_
