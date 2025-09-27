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
 * \file foreach_add_scalar_list_apt.cpp
 * \brief
 */
#include "./arch35/foreach_add_scalar_list_regbase.h"

using namespace ForeachAddScalarList;

template <typename T, typename ScalarT>
__aicore__ inline void ForeachAddScalarListImpl(
    GM_ADDR tensor1, GM_ADDR scalars, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachSoloTilingDataRegbase, tiling_data_in, tiling);
    const ForeachSoloTilingDataRegbase* __restrict tilingData = &tiling_data_in;
    ForeachAddScalarListRegbase<T, ScalarT, ForeachSoloTilingDataRegbase> op;
    op.Init(tensor1, scalars, outputs, workspace, tilingData, tPipe);
    op.Process();
}

extern "C" __global__ __aicore__ void foreach_add_scalar_list(
    GM_ADDR tensor1, GM_ADDR scalars, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachAddScalarListImpl<half, DTYPE_SCALARS>(tensor1, scalars, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachAddScalarListImpl<float, DTYPE_SCALARS>(tensor1, scalars, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachAddScalarListImpl<bfloat16_t, DTYPE_SCALARS>(tensor1, scalars, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT)) {
        ForeachAddScalarListImpl<int32_t, DTYPE_SCALARS>(tensor1, scalars, outputs, workspace, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}