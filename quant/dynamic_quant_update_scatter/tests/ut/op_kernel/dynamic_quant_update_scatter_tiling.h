/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _Dynamic_Quant_TILING_H_
#define _Dynamic_Quant_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define DTYPE_VAR int8_t
#define DTYPE_INDICES int32_t
#define DTYPE_UPDATES half
#define ORIG_DTYPE_UPDATES DT_FLOAT16

#define __CCE_UT_TEST__

#pragma pack(1)

struct DynamicQuantUpdateScatterTilingData {
    int64_t coreNum;           // ceil_div(updates_shape[0] * updates_shape[1] eachCoreBsNum
    int64_t eachCoreBsNum;     // ceil_div(updates_shape[0] * updates_shape[1] ori_core_num
    int64_t lastCoreBsNum;     // updates_shape[0] * updates_shape[1] - eachCoreBsNum * (core_num - 1
    int64_t updateAxisShape;   // updates_shape[axis]
    int64_t srcBsStride;       // updates_shape[2] * updates_shape[3]
    int64_t dstBsStride;       // data_shape[2] *  data_shape[3]
    int64_t indexElements;     // index_shape[0]
    int64_t numHead;           // data_shape[1]
    int64_t sizePerHead;       // not axis shape data_shape[2] or data_shape[3]
    int64_t dataAxisShape;     // data_shape of axis
    int64_t numOneBlock;       // 32 / dtype_size
    int64_t innerLoopEle;      // nums of ele per inner loop
    int64_t indicesShapeRank;  // rank of indices shape
    int64_t srcFirBsStride;    // updates_shape[1] * updates_shape[2] * updates_shape[3]
    int64_t dstFirSecBsStride; // data_shape[1] * data_shape[2] *  data_shape[3]
    int64_t updateDim0;        // updates_shape[0]
    int64_t updateDim1;        // updates_shape[1]
    int64_t varElements;
    int64_t varScalesElements;
    int64_t updatesElements;
    int64_t quantReptNum;
    int64_t varOrigLastDimSize;
    int64_t sizeSrcPerHead; // not axis shape data_shape[2] or data_shape[3]
    int64_t innerLoopFullRpt;
    int64_t innerLoopTimes; // inner loop time
    int64_t innerLoopTail;  // nums of ele tail inner loop
    int64_t innerLoopTimesLastCore;
    int64_t innerLoopTailLastCore;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                           \
    DynamicQuantUpdateScatterTilingData tilingData;                                          \
    INIT_TILING_DATA(DynamicQuantUpdateScatterTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreNum = tilingDataPointer->coreNum;                                       \
    (tilingData).eachCoreBsNum = tilingDataPointer->eachCoreBsNum;                           \
    (tilingData).lastCoreBsNum = tilingDataPointer->lastCoreBsNum;                           \
    (tilingData).updateAxisShape = tilingDataPointer->updateAxisShape;                       \
    (tilingData).srcBsStride = tilingDataPointer->srcBsStride;                               \
    (tilingData).dstBsStride = tilingDataPointer->dstBsStride;                               \
    (tilingData).indexElements = tilingDataPointer->indexElements;                           \
    (tilingData).numHead = tilingDataPointer->numHead;                                       \
    (tilingData).sizePerHead = tilingDataPointer->sizePerHead;                               \
    (tilingData).dataAxisShape = tilingDataPointer->dataAxisShape;                           \
    (tilingData).numOneBlock = tilingDataPointer->numOneBlock;                               \
    (tilingData).innerLoopEle = tilingDataPointer->innerLoopEle;                             \
    (tilingData).indicesShapeRank = tilingDataPointer->indicesShapeRank;                     \
    (tilingData).srcFirBsStride = tilingDataPointer->srcFirBsStride;                         \
    (tilingData).dstFirSecBsStride = tilingDataPointer->dstFirSecBsStride;                   \
    (tilingData).updateDim0 = tilingDataPointer->updateDim0;                                 \
    (tilingData).updateDim1 = tilingDataPointer->updateDim1;                                 \
    (tilingData).varElements = tilingDataPointer->varElements;                               \
    (tilingData).varScalesElements = tilingDataPointer->varScalesElements;                   \
    (tilingData).updatesElements = tilingDataPointer->updatesElements;                       \
    (tilingData).quantReptNum = tilingDataPointer->quantReptNum;                             \
    (tilingData).varOrigLastDimSize = tilingDataPointer->varOrigLastDimSize;                 \
    (tilingData).sizeSrcPerHead = tilingDataPointer->sizeSrcPerHead;                         \
    (tilingData).innerLoopFullRpt = tilingDataPointer->innerLoopFullRpt;                     \
    (tilingData).innerLoopTimes = tilingDataPointer->innerLoopTimes;                         \
    (tilingData).innerLoopTail = tilingDataPointer->innerLoopTail;                           \
    (tilingData).innerLoopTimesLastCore = tilingDataPointer->innerLoopTimesLastCore;         \
    (tilingData).innerLoopTailLastCore = tilingDataPointer->innerLoopTailLastCore;
#endif