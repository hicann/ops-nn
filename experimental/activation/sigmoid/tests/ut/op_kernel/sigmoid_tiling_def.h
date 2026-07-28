/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGMOID_TILING_DEF_H
#define SIGMOID_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define __CCE_UT_TEST__

struct SigmoidTilingData {
    int64_t dim0;
    int64_t blockFormer;
    int64_t ubFormer;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
};

inline void ISigmoidTilingData(uint8_t* tiling, SigmoidTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SigmoidTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    SigmoidTilingData tilingData;                  \
    ISigmoidTilingData(tilingPointer, &tilingData)
#endif
