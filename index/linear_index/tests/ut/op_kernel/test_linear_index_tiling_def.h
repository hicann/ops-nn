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
 * \file test_linear_index_tiling.h
 * \brief
 */
#ifndef _FAST_OP_TEST_LINEAR_INDEX_TILING_H_
#define _FAST_OP_TEST_LINEAR_INDEX_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct LinearIndexTilingDataDef {
    uint64_t usedCoreNum = 0;
    uint64_t eachCount = 0;
    uint64_t lastCount = 0;
    uint64_t indicesCount = 0;
    uint64_t indicesAlign = 0;
    uint64_t maxSize = 0;
    uint64_t eachNum = 0;
    uint64_t eachLoop = 0;
    uint64_t eachTail = 0;
    uint64_t lastNum = 0;
    uint64_t lastLoop = 0;
    uint64_t lastTail = 0;
    uint64_t target = 0;
    uint64_t selfStride = 0;
    uint64_t indicesStride = 0;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                             \
    LinearIndexTilingDataDef tilingData;                                          \
    INIT_TILING_DATA(LinearIndexTilingDataDef, tilingDataPointer, tilingPointer); \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                 \
    (tilingData).eachCount = tilingDataPointer->eachCount;                     \
    (tilingData).lastCount = tilingDataPointer->lastCount;                     \
    (tilingData).indicesCount = tilingDataPointer->indicesCount;               \
    (tilingData).indicesAlign = tilingDataPointer->indicesAlign;               \
    (tilingData).maxSize = tilingDataPointer->maxSize;                         \
    (tilingData).eachNum = tilingDataPointer->eachNum;                         \
    (tilingData).eachLoop = tilingDataPointer->eachLoop;                       \
    (tilingData).eachTail = tilingDataPointer->eachTail;                       \
    (tilingData).lastNum = tilingDataPointer->lastNum;                         \
    (tilingData).lastLoop = tilingDataPointer->lastLoop;                       \
    (tilingData).lastTail = tilingDataPointer->lastTail;                       \
    (tilingData).target = tilingDataPointer->target;                           \
    (tilingData).selfStride = tilingDataPointer->selfStride;                   \
    (tilingData).indicesStride = tilingDataPointer->indicesStride;
#endif // _FAST_OP_TEST_LINEAR_INDEX_TILING_H_
