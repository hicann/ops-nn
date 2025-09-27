/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TEST_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_TILING_H_
#define _TEST_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct MultiScaleDeformableAttentionGradTilingData {
  uint64_t batchSize = 0;
  uint64_t numKeys = 0;
  uint64_t numHeads = 0;
  uint64_t embedDims = 0;
  uint64_t numLevels = 0;
  uint64_t numQueries = 0;
  uint64_t numPoints = 0;
  uint64_t maxUbNum = 0;
  uint64_t coreNum = 0;
};

#define DTYPE_X int64_t
#define DTYPE_VALUE float
#define DTYPE_SPATIAL_SHAPES int32_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  __ubuf__ tilingStruct* tilingDataPointer =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                  \
  MultiScaleDeformableAttentionGradTilingData tilingData;                                         \
  INIT_TILING_DATA(MultiScaleDeformableAttentionGradTilingData, tilingDataPointer, tilingPointer);\
  (tilingData).batchSize = tilingDataPointer->batchSize;                            \
  (tilingData).numKeys = tilingDataPointer->numKeys;                                \
  (tilingData).numHeads = tilingDataPointer->numHeads;                              \
  (tilingData).embedDims = tilingDataPointer->embedDims;                            \
  (tilingData).numLevels = tilingDataPointer->numLevels;                            \
  (tilingData).numQueries = tilingDataPointer->numQueries;                          \
  (tilingData).numPoints = tilingDataPointer->numPoints;                            \
  (tilingData).maxUbNum = tilingDataPointer->maxUbNum;                            \
  (tilingData).coreNum = tilingDataPointer->coreNum;
#endif