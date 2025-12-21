/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _GE_GROUP_NORM_SILU_TILING_H_
#define _GE_GROUP_NORM_SILU_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define DTYPE_GAMMA float
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  GroupNormSiluTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupNormSiluTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).numGroups = tilingDataPointer->numGroups;                 \
  (tilingData).hwNum = tilingDataPointer->hwNum;                         \
  (tilingData).elemNum = tilingDataPointer->elemNum;                     \
  (tilingData).shapeC = tilingDataPointer->shapeC;                       \
  (tilingData).shapeD = tilingDataPointer->shapeD;                       \
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
