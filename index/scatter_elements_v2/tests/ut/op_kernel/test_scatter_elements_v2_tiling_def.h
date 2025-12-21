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
 * \file test_scatter_elements_v2_tiling.h
 * \brief
 */
#ifndef _FAST_OP_TEST_SCATTER_ELEMENTS_V2_TILING_H_
#define _FAST_OP_TEST_SCATTER_ELEMENTS_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct ScatterElementsV2TilingDataDef {
    uint32_t usedCoreNum = 0;
    uint64_t eachNum = 0;
    uint32_t extraTaskCore = 0;
    uint32_t eachPiece = 0;
    uint64_t inputOnePiece = 0;
    uint64_t inputCount = 0;
    uint64_t indicesCount = 0;
    uint64_t updatesCount = 0;
    uint64_t inputOneTime = 0;
    uint64_t indicesOneTime = 0;
    uint64_t updatesOneTime = 0;
    uint64_t inputEach = 0;
    uint64_t indicesEach = 0;
    uint64_t updatesEach = 0;
    uint64_t inputLast = 0;
    uint64_t indicesLast = 0;
    uint64_t updatesLast = 0;
    uint64_t inputLoop = 0;
    uint64_t indicesLoop = 0;
    uint64_t updatesLoop = 0;
    uint32_t inputAlign = 0;
    uint32_t indicesAlign = 0;
    uint32_t updatesAlign = 0;
    uint64_t lastIndicesLoop = 0;
    uint64_t lastIndicesEach = 0;
    uint64_t lastIndicesLast = 0;
    uint64_t oneTime = 0;
    uint64_t lastOneTime = 0;
    uint32_t modeFlag = 0;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                   \
    ScatterElementsV2TilingDataDef tilingData;                                          \
    INIT_TILING_DATA(ScatterElementsV2TilingDataDef, tilingDataPointer, tilingPointer); \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;                       \
    (tilingData).eachNum = tilingDataPointer->eachNum;                               \
    (tilingData).modeFlag = tilingDataPointer->modeFlag;                             \
    (tilingData).extraTaskCore = tilingDataPointer->extraTaskCore;                   \
    (tilingData).inputCount = tilingDataPointer->inputCount;                         \
    (tilingData).indicesCount = tilingDataPointer->indicesCount;                     \
    (tilingData).updatesCount = tilingDataPointer->updatesCount;                     \
    (tilingData).inputOneTime = tilingDataPointer->inputOneTime;                     \
    (tilingData).indicesOneTime = tilingDataPointer->indicesOneTime;                 \
    (tilingData).updatesOneTime = tilingDataPointer->updatesOneTime;                 \
    (tilingData).inputEach = tilingDataPointer->inputEach;                           \
    (tilingData).indicesEach = tilingDataPointer->indicesEach;                       \
    (tilingData).updatesEach = tilingDataPointer->updatesEach;                       \
    (tilingData).inputLast = tilingDataPointer->inputLast;                           \
    (tilingData).indicesLast = tilingDataPointer->indicesLast;                       \
    (tilingData).updatesLast = tilingDataPointer->updatesLast;                       \
    (tilingData).inputLoop = tilingDataPointer->inputLoop;                           \
    (tilingData).indicesLoop = tilingDataPointer->indicesLoop;                       \
    (tilingData).updatesLoop = tilingDataPointer->updatesLoop;                       \
    (tilingData).eachPiece = tilingDataPointer->eachPiece;                           \
    (tilingData).inputOnePiece = tilingDataPointer->inputOnePiece;                   \
    (tilingData).inputAlign = tilingDataPointer->inputAlign;                         \
    (tilingData).indicesAlign = tilingDataPointer->indicesAlign;                     \
    (tilingData).updatesAlign = tilingDataPointer->updatesAlign;                     \
    (tilingData).lastIndicesLoop = tilingDataPointer->lastIndicesLoop;               \
    (tilingData).lastIndicesEach = tilingDataPointer->lastIndicesEach;               \
    (tilingData).lastIndicesLast = tilingDataPointer->lastIndicesLast;               \
    (tilingData).oneTime = tilingDataPointer->oneTime;                               \
    (tilingData).lastOneTime = tilingDataPointer->lastOneTime;
#endif // _FAST_OP_TEST_SCATTER_ELEMENTS_V2_TILING_H_
