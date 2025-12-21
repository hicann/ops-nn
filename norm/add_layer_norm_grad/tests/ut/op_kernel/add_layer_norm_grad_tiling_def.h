/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_GRAD_TILING_H_
#define ADD_LAYER_NORM_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddLayerNormGradTilingData {
    uint32_t numCore = 0;
    uint32_t numLastDim = 0;
    uint32_t numFirstDim = 0;
    uint32_t nInOneCoreLength = 0;
    uint32_t nInOneCoreLengthTail = 0;
    uint32_t ndInOneCoreLength = 0;
    uint32_t nAvailInUb = 0;
    uint32_t dInnerLength = 0;
    uint32_t dInnerLengthTail = 0;
    uint32_t dOuterLength = 0;
    uint32_t nInOneCoreNorm = 0;
    uint32_t gmOneCoreElemXYNorm = 0;
    uint32_t nAvailInUbNorm = 0;
    uint32_t nMiddleCountNorm = 0;
    uint32_t ndRoundUpDtypeNorm = 0;
    uint32_t n1RoundUpFloatNorm = 0;
    uint32_t nInUbTotalNormTail = 0;
    uint32_t nInOneCoreTail = 0;
    uint32_t gmOneCoreElemXYTail = 0;
    uint32_t nAvailInUbTail = 0;
    uint32_t nMiddleCountTail = 0;
    uint32_t ndRoundUpDtypeTail = 0;
    uint32_t n1RoundUpFloatTail = 0;
    uint32_t nInUbTotalTailTail = 0;
    uint32_t dyPadRight = 0;
    uint32_t rstdPadRight = 0;
    uint32_t roundUpNumLastDim = 0;
    uint32_t roundUpNumLastDimDtype = 0;
    uint32_t roundUp1Dtype = 0;
    uint32_t roundUpNumLastDimFloat = 0;
    uint32_t workspace_size = 0;
    uint32_t isDeterministicKey = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    AddLayerNormGradTilingData tilingData;                                           \
    INIT_TILING_DATA(AddLayerNormGradTilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).numCore = tilingDataPointer->numCore;                               \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                         \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                       \
    (tilingData).nInOneCoreLength = tilingDataPointer->nInOneCoreLength;             \
    (tilingData).nInOneCoreLengthTail = tilingDataPointer->nInOneCoreLengthTail;     \
    (tilingData).ndInOneCoreLength = tilingDataPointer->ndInOneCoreLength;           \
    (tilingData).nAvailInUb = tilingDataPointer->nAvailInUb;                         \
    (tilingData).dInnerLength = tilingDataPointer->dInnerLength;                     \
    (tilingData).dInnerLengthTail = tilingDataPointer->dInnerLengthTail;             \
    (tilingData).dOuterLength = tilingDataPointer->dOuterLength;                     \
    (tilingData).nInOneCoreNorm = tilingDataPointer->nInOneCoreNorm;                 \
    (tilingData).gmOneCoreElemXYNorm = tilingDataPointer->gmOneCoreElemXYNorm;       \
    (tilingData).nAvailInUbNorm = tilingDataPointer->nAvailInUbNorm;                 \
    (tilingData).nMiddleCountNorm = tilingDataPointer->nMiddleCountNorm;             \
    (tilingData).ndRoundUpDtypeNorm = tilingDataPointer->ndRoundUpDtypeNorm;         \
    (tilingData).n1RoundUpFloatNorm = tilingDataPointer->n1RoundUpFloatNorm;         \
    (tilingData).nInUbTotalNormTail = tilingDataPointer->nInUbTotalNormTail;         \
    (tilingData).nInOneCoreTail = tilingDataPointer->nInOneCoreTail;                 \
    (tilingData).gmOneCoreElemXYTail = tilingDataPointer->gmOneCoreElemXYTail;       \
    (tilingData).nAvailInUbTail = tilingDataPointer->nAvailInUbTail;                 \
    (tilingData).nMiddleCountTail = tilingDataPointer->nMiddleCountTail;             \
    (tilingData).ndRoundUpDtypeTail = tilingDataPointer->ndRoundUpDtypeTail;         \
    (tilingData).n1RoundUpFloatTail = tilingDataPointer->n1RoundUpFloatTail;         \
    (tilingData).nInUbTotalTailTail = tilingDataPointer->nInUbTotalTailTail;         \
    (tilingData).dyPadRight = tilingDataPointer->dyPadRight;                         \
    (tilingData).rstdPadRight = tilingDataPointer->rstdPadRight;                     \
    (tilingData).roundUpNumLastDim = tilingDataPointer->roundUpNumLastDim;           \
    (tilingData).roundUpNumLastDimDtype = tilingDataPointer->roundUpNumLastDimDtype; \
    (tilingData).roundUp1Dtype = tilingDataPointer->roundUp1Dtype;                   \
    (tilingData).roundUpNumLastDimFloat = tilingDataPointer->roundUpNumLastDimFloat; \
    (tilingData).workspace_size = tilingDataPointer->workspace_size;                 \
    (tilingData).isDeterministicKey = tilingDataPointer->isDeterministicKey;
#endif