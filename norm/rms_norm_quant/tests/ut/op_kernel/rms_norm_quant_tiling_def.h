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
 * \file test_rms_norm_quant.h
 * \brief
 */
#define DTYPE_Y int

#ifndef _FAST_OP_TEST_RMS_NORM_ATB_H_
#define _FAST_OP_TEST_RMS_NORM_ATB_H_

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_KT_TEST__
#pragma pack(1)

struct NormCommonTilingData1 {
    uint32_t numCore{0};
    uint32_t numCol{0};
    uint32_t numRow{0};
    float avgFactor{0};
    float epsilon{0};
    uint32_t sliceSize{0};
    uint32_t sliceNum{0};
    uint32_t tailSliceSize{0};
    int32_t quantMin{0};
    float scale{0};
    int32_t offset{0};
    bool highPrecisionMode{false};
    bool gemmaMode{false};
    int32_t dstType{2};
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer))

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)

#define GET_TILING_DATA(tilingData, tilingPointer)                             \
    NormCommonTilingData1 tilingData;                                          \
    INIT_TILING_DATA(NormCommonTilingData1, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                         \
    (tilingData).numCol = tilingDataPointer->numCol;                           \
    (tilingData).numRow = tilingDataPointer->numRow;                           \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;                     \
    (tilingData).epsilon = tilingDataPointer->epsilon;                         \
    (tilingData).sliceSize = tilingDataPointer->sliceSize;                     \
    (tilingData).sliceNum = tilingDataPointer->sliceNum;                       \
    (tilingData).tailSliceSize = tilingDataPointer->tailSliceSize;             \
    (tilingData).quantMin = tilingDataPointer->quantMin;                       \
    (tilingData).scale = tilingDataPointer->scale;                             \
    (tilingData).offset = tilingDataPointer->offset;                           \
    (tilingData).highPrecisionMode = tilingDataPointer->highPrecisionMode;     \
    (tilingData).gemmaMode = tilingDataPointer->gemmaMode;                     \
    (tilingData).dstType = tilingDataPointer->dstType;

#endif // _FAST_OP_TEST_RMS_NORM_ATB_H_