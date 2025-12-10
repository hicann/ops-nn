/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEST_DEQUANT_SWIGLU_QUANT_H
#define TEST_DEQUANT_SWIGLU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define __aicore__

#pragma pack(1)

struct SwiGluTilingData {
    uint32_t is32BAligned;
    uint32_t isDoubleBuffer;
    uint64_t rowLen;
    uint64_t colLen;
    uint32_t baseRowLen;
    uint32_t baseColLen;
    uint32_t activateLeft;
    uint32_t biasIsEmpty;
    uint32_t quantScaleIsEmpty;
    uint32_t activateScaleIsEmpty;
    uint64_t swiColLen;
    uint64_t perRowLen;
    uint64_t modRowLen;
    uint32_t usedCoreNum;
};

#pragma pack()

struct DequantSwigluQuantBaseTilingData {
  int64_t inDimx;
  int64_t inDimy;
  int64_t outDimy;
  int64_t UbFactorDimx;
  int64_t UbFactorDimy;
  int64_t usedCoreNum;
  int64_t maxCoreNum;
  int64_t inGroupNum;
  int64_t quantMode;
  int64_t actRight;
  int64_t quantScaleDtype;
  int64_t groupIndexDtype;
  int64_t needSmoothScale;
  int64_t speGroupType;
  int64_t activationScaleIsEmpty;
  int64_t quantIsOne;
  int64_t swigluMode;
  float clampLimit;
  float gluAlpha;
  float gluBias;
};

inline void IDequantSwigluQuantTilingData(uint8_t* tiling, SwiGluTilingData* constData)
{
    memcpy(constData, tiling, sizeof(SwiGluTilingData));
}

inline void IDequantSwigluQuantTilingData(uint8_t* tiling, DequantSwigluQuantBaseTilingData* constData)
{
    memcpy(constData, tiling, sizeof(DequantSwigluQuantBaseTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg)      \
    tilingStruct tilingData;                                                  \
    IDequantSwigluQuantTilingData(tilingArg, &tilingData)

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
    __ubuf__ tilingStruct* tilingDataPointer =                                 \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    SwiGluTilingData tilingData;                                               \
    INIT_TILING_DATA(SwiGluTilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).rowLen = tilingDataPointer->rowLen;                              \
    (tilingData).colLen = tilingDataPointer->colLen;                          \
    (tilingData).is32BAligned = tilingDataPointer->is32BAligned;            \
    (tilingData).isDoubleBuffer = tilingDataPointer->isDoubleBuffer;              \
    (tilingData).baseRowLen = tilingDataPointer->baseRowLen;            \
    (tilingData).baseColLen = tilingDataPointer->baseColLen;            \
    (tilingData).activateLeft = tilingDataPointer->activateLeft;        \
    (tilingData).biasIsEmpty = tilingDataPointer->biasIsEmpty;                          \
    (tilingData).quantScaleIsEmpty = tilingDataPointer->quantScaleIsEmpty;            \
    (tilingData).activateScaleIsEmpty = tilingDataPointer->activateScaleIsEmpty;            \
    (tilingData).swiColLen = tilingDataPointer->swiColLen;              \
    (tilingData).perRowLen = tilingDataPointer->perRowLen;            \
    (tilingData).modRowLen = tilingDataPointer->modRowLen;            \
    (tilingData).usedCoreNum = tilingDataPointer->usedCoreNum;

#define DTYPE_QUANT_SCALE half

#endif
