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
 * \file scatter_list_tiling.h
 * \brief
 */
#ifndef SCATTER_LIST_TILING_H
#define SCATTER_LIST_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct ScatterListTilingData {
    int64_t dim0Count = 0;
    int64_t dim1Count = 0;
    int64_t varDim2Count = 0;
    int64_t dim2Count = 0;
    int64_t dim3Count = 0;
    int64_t dim3CountAlign = 0;
    int64_t updatesOneBlock = 0;
    int64_t indiceDims = 0;
    int64_t indiceCount = 0;
    int64_t indiceUbSize = 0;
    int64_t maskCount = 0;
    int64_t maskUbSize = 0;
    int64_t srcBatchStride = 0;
    int64_t srcBatchStrideAlign = 0;
    int64_t dstBatchStride = 0;
    int64_t useCoreNum = 0;
    int64_t preCoreBatchNum = 0;
    int64_t lastCoreBatchNum = 0;
    int64_t eachLoopNum = 0;
    int64_t eachPreLoopEle = 0;
    int64_t eachLastLoopEle = 0;
    int64_t eachLastLoopEleAlign = 0;
    int64_t updatesCount = 0;
    int64_t updatesUbSize = 0;
    int64_t dataUbSize = 0;
    int64_t transposeUbSize = 0;
    int64_t transRepeatTimes = 0;
    int64_t transRepeatTimesTail = 0;
    int64_t updateDim23Align = 0;
    int64_t preCoreUpdateDim23 = 0;
    int64_t varDim3Stride = 0;
    int64_t varDim3Count = 0;
    int64_t dim3CountSize = 0;
    int64_t eachLastSize = 0;
    int64_t tilingKey = 0;
};

#define DTYPE_VAR int8_t
#define DTYPE_INDICE int32_t

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                               \
    ScatterListTilingData tilingData;                                            \
    INIT_TILING_DATA(ScatterListTilingData, tilingDataPointer, tilingPointer);   \
    (tilingData).dim0Count = tilingDataPointer->dim0Count;                       \
    (tilingData).dim1Count = tilingDataPointer->dim1Count;                       \
    (tilingData).varDim2Count = tilingDataPointer->varDim2Count;                 \
    (tilingData).dim2Count = tilingDataPointer->dim2Count;                       \
    (tilingData).dim3Count = tilingDataPointer->dim3Count;                       \
    (tilingData).dim3CountAlign = tilingDataPointer->dim3CountAlign;             \
    (tilingData).updatesOneBlock = tilingDataPointer->updatesOneBlock;           \
    (tilingData).indiceDims = tilingDataPointer->indiceDims;                     \
    (tilingData).indiceCount = tilingDataPointer->indiceCount;                   \
    (tilingData).indiceUbSize = tilingDataPointer->indiceUbSize;                 \
    (tilingData).maskCount = tilingDataPointer->maskCount;                       \
    (tilingData).maskUbSize = tilingDataPointer->maskUbSize;                     \
    (tilingData).srcBatchStride = tilingDataPointer->srcBatchStride;             \
    (tilingData).srcBatchStrideAlign = tilingDataPointer->srcBatchStrideAlign;   \
    (tilingData).dstBatchStride = tilingDataPointer->dstBatchStride;             \
    (tilingData).useCoreNum = tilingDataPointer->useCoreNum;                     \
    (tilingData).preCoreBatchNum = tilingDataPointer->preCoreBatchNum;           \
    (tilingData).lastCoreBatchNum = tilingDataPointer->lastCoreBatchNum;         \
    (tilingData).eachLoopNum = tilingDataPointer->eachLoopNum;                   \
    (tilingData).eachPreLoopEle = tilingDataPointer->eachPreLoopEle;             \
    (tilingData).eachLastLoopEle = tilingDataPointer->eachLastLoopEle;           \
    (tilingData).eachLastLoopEleAlign = tilingDataPointer->eachLastLoopEleAlign; \
    (tilingData).updatesCount = tilingDataPointer->updatesCount;                 \
    (tilingData).updatesUbSize = tilingDataPointer->updatesUbSize;               \
    (tilingData).dataUbSize = tilingDataPointer->dataUbSize;                     \
    (tilingData).transposeUbSize = tilingDataPointer->transposeUbSize;           \
    (tilingData).transRepeatTimes = tilingDataPointer->transRepeatTimes;         \
    (tilingData).transRepeatTimesTail = tilingDataPointer->transRepeatTimesTail; \
    (tilingData).updateDim23Align = tilingDataPointer->updateDim23Align;         \
    (tilingData).preCoreUpdateDim23 = tilingDataPointer->preCoreUpdateDim23;     \
    (tilingData).varDim3Stride = tilingDataPointer->varDim3Stride;               \
    (tilingData).varDim3Count = tilingDataPointer->varDim3Count;                 \
    (tilingData).dim3CountSize = tilingDataPointer->dim3CountSize;               \
    (tilingData).eachLastSize = tilingDataPointer->eachLastSize;                 \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;
#endif // SCATTER_LIST_TILING_H
