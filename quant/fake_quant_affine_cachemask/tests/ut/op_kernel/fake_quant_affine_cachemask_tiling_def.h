/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FAST_OP_TEST_FAKE_QUANT_AFFINE_CACHEMASK_TILING_DEF_H_
#define _FAST_OP_TEST_FAKE_QUANT_AFFINE_CACHEMASK_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct FakeQuantAffineCachemaskTilingData {
    int64_t quantMin = 0;
    int64_t quantMax = 0;
    uint32_t loopNum = 0;
    uint32_t remainNum = 0;
    uint32_t calcLength = 0;
    uint32_t headNum = 0;
    uint32_t dataPerRepeat = 0;
    uint32_t totalLengthAligned = 0;
    uint32_t tileLength = 0;
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                          \
    FakeQuantAffineCachemaskTilingData tilingData;                                          \
    INIT_TILING_DATA(FakeQuantAffineCachemaskTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).quantMin = tilingDataPointer->quantMin;                                    \
    (tilingData).quantMax = tilingDataPointer->quantMax;                                    \
    (tilingData).loopNum = tilingDataPointer->loopNum;                                      \
    (tilingData).remainNum = tilingDataPointer->remainNum;                                  \
    (tilingData).calcLength = tilingDataPointer->calcLength;                                \
    (tilingData).headNum = tilingDataPointer->headNum;                                      \
    \
  (tilingData)                                                                              \
        .dataPerRepeat = tilingDataPointer->dataPerRepeat;                                  \
    (tilingData).totalLengthAligned = tilingDataPointer->totalLengthAligned;                \
    \
  (tilingData)                                                                              \
        .tileLength = tilingDataPointer->tileLength;
#endif // _FAST_OP_TEST_FAKE_QUANT_AFFINE_CACHEMASK_TILING_DEF_H_
