/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _GE_GROUP_NORM_SWISH_TILING_H_
#define _GE_GROUP_NORM_SWISH_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                           \
  GroupNormSwishTilingData tilingData;                                          \
  INIT_TILING_DATA(GroupNormSwishTilingData, tilingDataPointer, tilingPointer); \
  (tilingData).numGroups = tilingDataPointer->numGroups;                 \
  (tilingData).epsilon = tilingDataPointer->epsilon;                         \
  (tilingData).activateSwish = tilingDataPointer->activateSwish;                     \
  (tilingData).swishScale = tilingDataPointer->swishScale;                       \
  (tilingData).hwNum = tilingDataPointer->hwNum;                       \
  (tilingData).shapeC = tilingDataPointer->shapeC;             \
  (tilingData).shapeCAlign = tilingDataPointer->shapeCAlign;               \
  (tilingData).shapeD = tilingDataPointer->shapeD;             \
  (tilingData).numPerGroup = tilingDataPointer->numPerGroup;             \
  (tilingData).groupPerCore = tilingDataPointer->groupPerCore;                     \
  (tilingData).groupLastCore = tilingDataPointer->groupLastCore;                   \
  (tilingData).groupPerCoreAlign = tilingDataPointer->groupPerCoreAlign;           \
  (tilingData).numPerLoop = tilingDataPointer->numPerLoop;         \
  (tilingData).loopTimes = tilingDataPointer->loopTimes;                 \
  (tilingData).loopTimesAlign = tilingDataPointer->loopTimesAlign;                 \
  (tilingData).numTailLoop = tilingDataPointer->numTailLoop;

#endif
