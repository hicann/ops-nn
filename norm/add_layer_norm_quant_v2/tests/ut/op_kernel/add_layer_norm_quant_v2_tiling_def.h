/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_QUANT_V2_TILING_DEF_H_
#define ADD_LAYER_NORM_QUANT_V2_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddLayerNormQuantV2TilingData {
    uint32_t numCore = 0;
    uint32_t numLastDim = 0;
    uint32_t numFirstDim = 0;
    uint32_t firstDimPerCore = 0;
    uint32_t firstDimPerCoreTail = 0;
    uint32_t numLastDimAlign32 = 0;
    uint32_t firstDimPerTime = 0;
    uint32_t lastDimPerTime = 0;
    float eps = 0;
    float aveFactor = 0;
    uint32_t colMoveCnt = 0;
    uint32_t sliceSize = 0;
    uint32_t colTail = 0;
    uint32_t isXOut = 0;
    uint32_t scaleOffsetMode = 0;
    uint32_t isPerTensor = 0;
    uint32_t sliceNum = 0;
    uint32_t workspaceSize = 0;
    uint32_t numLastDimAlign = 0;
    uint32_t tailSliceSize = 0;
    uint32_t mulLoopFp32 = 0;
    uint32_t mulTailFp32 = 0;
    uint8_t dstRepStrideFp32 = 0;
    uint32_t firstDimPerTimeTail = 0;
    uint32_t rowTailPerBlock = 0;
    uint32_t rowTailLastBlock = 0;
    uint64_t gmOffset = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                      \
    AddLayerNormQuantV2TilingData tilingData;                                           \
    INIT_TILING_DATA(AddLayerNormQuantV2TilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).numCore = tilingDataPointer->numCore;                                  \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                            \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                          \
    (tilingData).firstDimPerCore = tilingDataPointer->firstDimPerCore;                  \
    (tilingData).firstDimPerCoreTail = tilingDataPointer->firstDimPerCoreTail;          \
    (tilingData).numLastDimAlign32 = tilingDataPointer->numLastDimAlign32;              \
    (tilingData).firstDimPerTime = tilingDataPointer->firstDimPerTime;                  \
    (tilingData).lastDimPerTime = tilingDataPointer->lastDimPerTime;                    \
    (tilingData).eps = tilingDataPointer->eps;                                          \
    (tilingData).aveFactor = tilingDataPointer->aveFactor;                              \
    (tilingData).colMoveCnt = tilingDataPointer->colMoveCnt;                            \
    (tilingData).sliceSize = tilingDataPointer->sliceSize;                              \
    (tilingData).colTail = tilingDataPointer->colTail;                                  \
    (tilingData).isXOut = tilingDataPointer->isXOut;                                    \
    (tilingData).scaleOffsetMode = tilingDataPointer->scaleOffsetMode;                  \
    (tilingData).isPerTensor = tilingDataPointer->isPerTensor;                          \
    (tilingData).sliceNum = tilingDataPointer->sliceNum;                                \
    (tilingData).workspaceSize = tilingDataPointer->workspaceSize;                      \
    (tilingData).numLastDimAlign = tilingDataPointer->numLastDimAlign;                  \
    (tilingData).tailSliceSize = tilingDataPointer->tailSliceSize;                      \
    (tilingData).mulLoopFp32 = tilingDataPointer->mulLoopFp32;                          \
    (tilingData).mulTailFp32 = tilingDataPointer->mulTailFp32;                          \
    (tilingData).dstRepStrideFp32 = tilingDataPointer->dstRepStrideFp32;                \
    (tilingData).firstDimPerTimeTail = tilingDataPointer->firstDimPerTimeTail;          \
    (tilingData).rowTailPerBlock = tilingDataPointer->rowTailPerBlock;                  \
    (tilingData).rowTailLastBlock = tilingDataPointer->rowTailLastBlock;                \
    (tilingData).gmOffset = tilingDataPointer->gmOffset;
#endif
