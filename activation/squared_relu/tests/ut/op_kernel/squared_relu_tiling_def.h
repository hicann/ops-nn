/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SQUARED_RELU_TILING_DEF_H
#define SQUARED_RELU_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

// #define DT_BF16 bfloat16_t
// #define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__


#define __aicore__

struct SquaredReluTilingData {
  int32_t elementNum = 8;
  uint32_t needCoreNum = 1;
};

inline void ISquaredReluTilingData(uint8_t* tiling, SquaredReluTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SquaredReluTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  SquaredReluTilingData tilingData;                           \
  ISquaredReluTilingData(tilingPointer, &tilingData)
#endif