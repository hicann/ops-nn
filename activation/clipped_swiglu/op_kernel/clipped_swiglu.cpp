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
 * \file clipped_swiglu.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "clipped_swiglu.hpp"

using namespace AscendC;
extern "C" __global__ __aicore__ void clipped_swiglu(
    GM_ADDR xGM, GM_ADDR groupIndexGM, GM_ADDR yGM, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userspace = GetUserWorkspace(workspace);
    if (userspace == nullptr) {
        return;
    }
    TPipe pipe;
#if (ORIG_DTYPE_X == DT_FLOAT)
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(ClippedSwigluTilingData, tilingDataIn, tiling);
        const ClippedSwigluTilingData* __restrict__ tilingData = &tilingDataIn;
        ClippedSwigluOps::ClippedSwigluBase<float> op(&pipe);
        op.Init(xGM, groupIndexGM, yGM, tilingData);
        op.Process();
    }
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(ClippedSwigluTilingData, tilingDataIn, tiling);
        const ClippedSwigluTilingData* __restrict__ tilingData = &tilingDataIn;
        ClippedSwigluOps::ClippedSwigluBase<half> op(&pipe);
        op.Init(xGM, groupIndexGM, yGM, tilingData);
        op.Process();
    }
#endif
#if (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(ClippedSwigluTilingData, tilingDataIn, tiling);
        const ClippedSwigluTilingData* __restrict__ tilingData = &tilingDataIn;
        ClippedSwigluOps::ClippedSwigluBase<bfloat16_t> op(&pipe);
        op.Init(xGM, groupIndexGM, yGM, tilingData);
        op.Process();
    }
#endif
}