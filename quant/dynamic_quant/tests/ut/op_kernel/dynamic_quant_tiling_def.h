/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _Dynamic_Quant_TILING_H_
#define _Dynamic_Quant_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define DTYPE_X bfloat16_t
#define ORIG_DTYPE_Y DT_INT4
#define DTYPE_Y int4b_t
#define __CCE_UT_TEST__

#pragma pack(1)

struct DynamicQuantTilingData {
    uint32_t coreNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNumHeadCore;
    uint32_t multiRowNumTailCore;
    uint32_t innerLoopEle;
    uint32_t innerLoopTimes;
    uint32_t innerLoopTail;
    uint32_t groupNum;
    uint32_t alignGroupNum;
    uint32_t hasSmooth;

    uint32_t sizeH;
    uint32_t sizeX;
    uint32_t sizeZOut;
    uint32_t sizeCopyRow;
    uint32_t numCopyRow;
    uint32_t numHeadCore;
    uint32_t numTailCore;
    uint32_t numHeadTimes;
    uint32_t numTailTimes;
    uint32_t numLastTailRow;
    uint32_t alignType;
    uint32_t ubSize;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                  \
    DynamicQuantTilingData tilingData;                                          \
    INIT_TILING_DATA(DynamicQuantTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).coreNum = tilingDataPointer->coreNum;                          \
    (tilingData).rowLen = tilingDataPointer->rowLen;                            \
    (tilingData).headCoreNum = tilingDataPointer->headCoreNum;                  \
    (tilingData).rowPerHeadCore = tilingDataPointer->rowPerHeadCore;            \
    (tilingData).rowPerTailCore = tilingDataPointer->rowPerTailCore;            \
    (tilingData).multiRowNumHeadCore = tilingDataPointer->multiRowNumHeadCore;  \
    (tilingData).multiRowNumTailCore = tilingDataPointer->multiRowNumTailCore;  \
    (tilingData).innerLoopEle = tilingDataPointer->innerLoopEle;                \
    (tilingData).innerLoopTimes = tilingDataPointer->innerLoopTimes;            \
    (tilingData).innerLoopTail = tilingDataPointer->innerLoopTail;              \
    (tilingData).groupNum = tilingDataPointer->groupNum;                        \
    (tilingData).alignGroupNum = tilingDataPointer->alignGroupNum;              \
    (tilingData).hasSmooth = tilingDataPointer->hasSmooth;                      \
    (tilingData).sizeH = tilingDataPointer->sizeH;                              \
    (tilingData).sizeX = tilingDataPointer->sizeX;                              \
    (tilingData).sizeZOut = tilingDataPointer->sizeZOut;                        \
    (tilingData).sizeCopyRow = tilingDataPointer->sizeCopyRow;                  \
    (tilingData).numCopyRow = tilingDataPointer->numCopyRow;                    \
    (tilingData).numHeadCore = tilingDataPointer->numHeadCore;                  \
    (tilingData).numTailCore = tilingDataPointer->numTailCore;                  \
    (tilingData).numHeadTimes = tilingDataPointer->numHeadTimes;                \
    (tilingData).numTailTimes = tilingDataPointer->numTailTimes;                \
    (tilingData).numLastTailRow = tilingDataPointer->numLastTailRow;            \
    (tilingData).alignType = tilingDataPointer->alignType;                      \
    (tilingData).ubSize = tilingDataPointer->ubSize;
#endif