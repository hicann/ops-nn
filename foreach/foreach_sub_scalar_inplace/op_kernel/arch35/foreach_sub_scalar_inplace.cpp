/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file foreach_sub_scalar_inplace.cpp
 * \brief
 */
#include "foreach_sub_scalar_inplace_regbase.h"

using namespace ForeachSubScalarInplace;

template <typename T, typename ScalarT>
__aicore__ inline void ForeachSubScalarInplaceImpl(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachSoloTilingDataRegbase, tiling_data_in, tiling);
    const ForeachSoloTilingDataRegbase* __restrict tilingData = &tiling_data_in;
    ForeachSubScalarInplaceRegbase<T, ScalarT, ForeachSoloTilingDataRegbase> op;
    // inplace: feed x as both inputs and outputs
    op.Init(inputs, scalar, inputs, workspace, tilingData, tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void foreach_sub_scalar_inplace(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachSubScalarInplaceImpl<half, DTYPE_SCALAR>(x, scalar, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachSubScalarInplaceImpl<float, float>(x, scalar, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT)) {
        ForeachSubScalarInplaceImpl<int, int>(x, scalar, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachSubScalarInplaceImpl<bfloat16_t, float>(x, scalar, workspace, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}
