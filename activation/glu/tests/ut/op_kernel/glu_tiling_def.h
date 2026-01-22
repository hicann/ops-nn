/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _GLU_TILING_DEF_H_
#define _GLU_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DT_FP16 half
#define DT_FP32 float
#define DTYPE_X DT_FP32
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct GluTilingData {
    int64_t group;
    int64_t loopNum;
    int64_t tailLoopNum;
    int64_t nLastTailGroup;
    int64_t lastTailGroup;
    int64_t splitSize;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t tilingKey;
    int64_t blockSize;
    int64_t ny;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                         \
    GluTilingData tilingData;                                          \
    INIT_TILING_DATA(GluTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).group = tilingDataPointer->group;                         \
    (tilingData).loopNum = tilingDataPointer->loopNum;                     \
    (tilingData).tailLoopNum = tilingDataPointer->tailLoopNum;             \
    (tilingData).nLastTailGroup = tilingDataPointer->nLastTailGroup;       \
    (tilingData).lastTailGroup = tilingDataPointer->lastTailGroup;         \
    (tilingData).ny = tilingDataPointer->ny;                               \
    (tilingData).splitSize = tilingDataPointer->splitSize;                 \
    (tilingData).realCoreNum = tilingDataPointer->realCoreNum;             \
    (tilingData).numPerCore = tilingDataPointer->numPerCore;               \
    (tilingData).tilingKey = tilingDataPointer->tilingKey;                 \
    (tilingData).blockSize = tilingDataPointer->blockSize;                 
#endif