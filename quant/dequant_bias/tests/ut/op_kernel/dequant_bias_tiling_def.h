/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_DEQUANT_BIAS_H
#define TEST_DEQUANT_BIAS_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

// #define DT_BF16 bfloat16_t
// #define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif
#ifndef DTYPE_X
#define DTYPE_X int
#endif
#ifndef DTYPE_Y
#define DTYPE_Y int
#endif
#ifndef DTYPE_WEIGHT_SCALE
#define DTYPE_WEIGHT_SCALE int
#endif

#define __aicore__

struct DequantBiasTilingData {
  int64_t M = 40;
  int64_t N = 256;
  int64_t nAlign = 256;
  int64_t asExist = 1;
  int64_t needCoreNum = 40;
  int64_t perCoreRow = 1;
  int64_t tailCoreRow = 1;
  int64_t inBufferSize = 2048;
  int64_t wsBufferSize = 1024;
  int64_t asBufferSize = 32;
  int64_t biasBufferSize = 1024;
  int64_t perCoreLoopRow = 1;
  int64_t perCoreTailLoopRow = 1;
  int64_t perCoreLoops = 1;
  int64_t tailCoreLoopRow = 1;
  int64_t tailCoreTailLoopRow = 1;
  int64_t tailCoreLoops = 1;
};


inline void IDequantBiasTilingData(uint8_t* tiling, DequantBiasTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(DequantBiasTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  DequantBiasTilingData tilingData;                           \
  IDequantBiasTilingData(tilingPointer, &tilingData)
#endif