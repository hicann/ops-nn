/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_scatter_with_position.cpp
 * \brief
 */

#include "masked_scatter_with_position.h"

#define BA_INT32_TILING_KEY 100
#define BA_INT64_TILING_KEY 101
#define AB_INT32_TILING_KEY 200
#define AB_INT64_TILING_KEY 201
using namespace MaskedScatterWithPosition;

extern "C" __global__ __aicore__ void masked_scatter_with_position(
    GM_ADDR x, GM_ADDR mask, GM_ADDR position, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MaskedScatterWithPositionTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(BA_INT32_TILING_KEY)) {
        MaskedScatterWithPosition::MaskedScatterWithPositionSimt<DTYPE_Y, uint32_t, PATTERN_BA> op(&tilingData);
        op.Init(x, mask, position, updates, y);
        op.Process();
    } else if (TILING_KEY_IS(BA_INT64_TILING_KEY)) {
        MaskedScatterWithPosition::MaskedScatterWithPositionSimt<DTYPE_Y, uint64_t, PATTERN_BA> op(&tilingData);
        op.Init(x, mask, position, updates, y);
        op.Process();
    } else if (TILING_KEY_IS(AB_INT32_TILING_KEY)) {
        MaskedScatterWithPosition::MaskedScatterWithPositionSimt<DTYPE_Y, uint32_t, PATTERN_AB> op(&tilingData);
        op.Init(x, mask, position, updates, y);
        op.Process();
    } else if (TILING_KEY_IS(AB_INT64_TILING_KEY)) {
        MaskedScatterWithPosition::MaskedScatterWithPositionSimt<DTYPE_Y, uint64_t, PATTERN_AB> op(&tilingData);
        op.Init(x, mask, position, updates, y);
        op.Process();
    }
}