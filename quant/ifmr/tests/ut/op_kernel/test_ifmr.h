/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_IFMR_H
#define TEST_IFMR_H

#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)    \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                          \
  IfmrTilingData tilingData;                                                \
  INIT_TILING_DATA(IfmrTilingData, tilingDataPointer, tilingPointer);       \
  (tilingData).minPercentile = tilingDataPointer->minPercentile;            \
  (tilingData).maxPercentile = tilingDataPointer->maxPercentile;            \
  (tilingData).searchRange[0] = tilingDataPointer->searchRange[0];          \
  (tilingData).searchRange[1] = tilingDataPointer->searchRange[1];          \
  (tilingData).searchStep = tilingDataPointer->searchStep;                  \
  (tilingData).withOffset = tilingDataPointer->withOffset;                  \
  (tilingData).quantBits = tilingDataPointer->quantBits;                    \
  (tilingData).dataLength = tilingDataPointer->dataLength;                  \
  (tilingData).cumsumLength = tilingDataPointer->cumsumLength;
#endif // TEST_IFMR_H
