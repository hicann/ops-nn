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
 * \file foreach_lerp_scalar_apt.cpp
 * \brief
 */
#include "./arch35/foreach_lerp_scalar_regbase.h"

using namespace ForeachLerpScalar;

template <typename T>
__aicore__ inline void ForeachLerpScalarImpl(
    GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR weight, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachSoloTilingDataRegbase, tiling_data_in, tiling);
    const ForeachSoloTilingDataRegbase* __restrict tilingData = &tiling_data_in;
    ForeachLerpScalarRegbase<T, ForeachSoloTilingDataRegbase> op;
    op.Init(tensor1, tensor2, weight, outputs, workspace, tilingData, tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void foreach_lerp_scalar(
    GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR weight, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachLerpScalarImpl<half>(tensor1, tensor2, weight, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachLerpScalarImpl<float>(tensor1, tensor2, weight, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachLerpScalarImpl<bfloat16_t>(tensor1, tensor2, weight, outputs, workspace, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}