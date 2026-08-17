/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clipped_swiglu_grad.cpp
 * \brief Kernel entry for ClippedSwigluGrad (910B / 910_93)
 *
 * 使用 TPL tiling key 编译期特化 isInterleaved/isGroup
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "clipped_swiglu_grad_tiling_key.h"
#include "clipped_swiglu_grad.h"

using namespace AscendC;
using namespace ClippedSwigluGradArch35Op;

template <uint64_t isInterleaved, uint64_t isGroup>
__global__ __aicore__ void clipped_swiglu_grad(GM_ADDR gradYGM, GM_ADDR xGM, GM_ADDR groupIndexGM, GM_ADDR gradXOutGM,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ClippedSwigluGradTilingData, tilingData, tiling);
    TPipe pipe;

    if constexpr (isInterleaved == 1) {
        if constexpr (isGroup == 1) {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, true, true> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        } else {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, true, false> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        }
    } else {
        if constexpr (isGroup == 1) {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, false, true> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        } else {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, false, false> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        }
    }
}
