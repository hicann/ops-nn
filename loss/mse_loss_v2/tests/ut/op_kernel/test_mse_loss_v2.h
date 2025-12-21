/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _CROSS_ENTROPY_LOSS_GRAD_TILING_H_
#define _CROSS_ENTROPY_LOSS_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct MSELossV2TilingDataTest {
    uint64_t coreNum = 0;
    uint64_t tailElems = 0;
    uint64_t bufferNum = 0;
    float scale = 0;
    uint64_t epochs = 0;
    uint64_t epochsForLastCore = 0;
    uint64_t coreLength = 0;
    uint64_t tileLength = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailTileLengthForLastCore = 0;
};

inline void InitMSELossV2TilingData(uint8_t* tiling, MSELossV2TilingDataTest* const_data)
{
    memcpy(const_data, tiling, sizeof(MSELossV2TilingDataTest));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    MSELossV2TilingDataTest tilingData;                \
    InitMSELossV2TilingData(tilingPointer, &tilingData)
#endif