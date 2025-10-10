/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cross_v2.h
 * \brief
 */

#ifndef _TEST_CROSS_V2_H_
#define _TEST_CROSS_V2_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct CrossV2TilingTest {
    uint64_t stepSize;
    uint64_t tileNum;
    uint64_t tileNumPerBatch;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
};

inline void InitCrossV2TilingData(uint8_t* tiling, CrossV2TilingTest* const_data)
{
    memcpy(const_data, tiling, sizeof(CrossV2TilingTest));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    CrossV2TilingTest tilingData;                  \
    InitCrossV2TilingData(tilingPointer, &tilingData)
#endif
