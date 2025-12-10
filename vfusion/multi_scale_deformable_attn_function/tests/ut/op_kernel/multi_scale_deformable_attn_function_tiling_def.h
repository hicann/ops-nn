/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef _TEST_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_TILING_H_
#define _TEST_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct MultiScaleDeformableAttnFunctionTilingData {
  uint64_t batchSize = 0;
  uint64_t numKeys = 0;
  uint64_t numHeads = 0;
  uint64_t embedDims = 0;
  uint64_t numLevels = 0;
  uint64_t numQueries = 0;
  uint64_t numPoints = 0;
  uint64_t coreNum = 0;
  uint64_t pointLoops = 0;
  uint64_t realLevels = 0;
};

#define DTYPE_VALUE float
#define DTYPE_VALUE_SPATIAL_SHAPES int32_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                  \
  MultiScaleDeformableAttnFunctionTilingData tilingData;                                         \
  INIT_TILING_DATA(MultiScaleDeformableAttnFunctionTilingData, tilingDataPointer, tilingPointer);\
  (tilingData).batchSize = tilingDataPointer->batchSize;                            \
  (tilingData).numKeys = tilingDataPointer->numKeys;                                \
  (tilingData).numHeads = tilingDataPointer->numHeads;                              \
  (tilingData).embedDims = tilingDataPointer->embedDims;                            \
  (tilingData).numLevels = tilingDataPointer->numLevels;                            \
  (tilingData).numQueries = tilingDataPointer->numQueries;                          \
  (tilingData).numPoints = tilingDataPointer->numPoints;                            \
  (tilingData).coreNum = tilingDataPointer->coreNum;                                \
  (tilingData).pointLoops = tilingDataPointer->pointLoops;                          \
  (tilingData).realLevels = tilingDataPointer->realLevels;
#endif