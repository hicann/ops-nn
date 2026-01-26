/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _UT_GROUP_NORM_SILU_QUANT_TILING_H_
#define _UT_GROUP_NORM_SILU_QUANT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#pragma pack(1)

struct GroupNormSiluQuantTilingData {
  int64_t numGroups;
  int64_t hwNum;
  int64_t elemNum;
  int64_t shapeC;
  int64_t shapeD;
  int64_t shapeQuantScale;
  int64_t realCoreNum;
  int64_t numPerCore;
  int64_t numLastCore;
  int64_t processSize;
  int64_t loopNum;
  int64_t loopTail;
  int64_t innerLoopNum;
  int64_t innerLoopTail;
  int64_t tilingKey;
  float epsilon;
  int64_t activateSilu;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  GroupNormSiluQuantTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupNormSiluQuantTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).numGroups = tilingDataPointer->numGroups;                 \
  (tilingData).hwNum = tilingDataPointer->hwNum;                         \
  (tilingData).elemNum = tilingDataPointer->elemNum;                     \
  (tilingData).shapeC = tilingDataPointer->shapeC;                       \
  (tilingData).shapeD = tilingDataPointer->shapeD;                       \
  (tilingData).shapeQuantScale = tilingDataPointer->shapeQuantScale;                       \
  (tilingData).realCoreNum = tilingDataPointer->realCoreNum;             \
  (tilingData).numPerCore = tilingDataPointer->numPerCore;               \
  (tilingData).numLastCore = tilingDataPointer->numLastCore;             \
  (tilingData).processSize = tilingDataPointer->processSize;             \
  (tilingData).loopNum = tilingDataPointer->loopNum;                     \
  (tilingData).loopTail = tilingDataPointer->loopTail;                   \
  (tilingData).innerLoopNum = tilingDataPointer->innerLoopNum;           \
  (tilingData).innerLoopTail = tilingDataPointer->innerLoopTail;         \
  (tilingData).tilingKey = tilingDataPointer->tilingKey;                 \
  (tilingData).epsilon = tilingDataPointer->epsilon;                 \
  (tilingData).activateSilu = tilingDataPointer->activateSilu;

#endif
