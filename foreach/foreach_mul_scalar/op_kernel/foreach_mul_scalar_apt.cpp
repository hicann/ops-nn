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
 * \file foreach_mul_scalar_apt.cpp
 * \brief
 */
#include "./arch35/foreach_mul_scalar_regbase.h"

using namespace ForeachMulScalar;

template <typename T, typename ScalarT>
__aicore__ inline void ForeachMulScalarImpl(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachSoloTilingDataRegbase, tiling_data_in, tiling);
    const ForeachSoloTilingDataRegbase* __restrict tilingData = &tiling_data_in;
    ForeachMulScalarRegbase<T, ScalarT, ForeachSoloTilingDataRegbase> op;
    op.Init(inputs, scalar, outputs, workspace, tilingData, tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void foreach_mul_scalar(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachMulScalarImpl<half, DTYPE_SCALAR>(inputs, scalar, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachMulScalarImpl<float, float>(inputs, scalar, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT)) {
        ForeachMulScalarImpl<int, int>(inputs, scalar, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachMulScalarImpl<bfloat16_t, float>(inputs, scalar, outputs, workspace, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}