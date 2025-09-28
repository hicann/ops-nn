/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_FATRELU_MUL_H
#define TEST_FATRELU_MUL_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define DT_FP16 half
#define DT_FP32 float
#define DTYPE_INPUT DT_FP32

#define __CCE_UT_TEST__

#define __aicore__

struct HeavisideTilingData {
    int32_t elementNum = 8;
    bool valuesType = false;
    uint32_t needCoreNum = 1;
};

inline void IHeavisideTilingData(uint8_t* tiling, HeavisideTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(HeavisideTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    HeavisideTilingData tilingData;                \
    IHeavisideTilingData(tilingPointer, &tilingData)
#endif