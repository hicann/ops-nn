/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_GELU_MUL_H
#define TEST_GELU_MUL_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

#define __aicore__

struct GeluMulTilingData {
  int32_t lastDimSize = 4;
  int32_t batchSize = 2;
  uint32_t needCoreNum = 1;
  int32_t approximateMode = 0;
  int32_t PPMaxCalNum = 6144;
};

inline void IGeluMulTilingData(uint8_t* tiling, GeluMulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GeluMulTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
  GeluMulTilingData tilingData;                           \
  IGeluMulTilingData(tilingPointer, &tilingData)
#endif