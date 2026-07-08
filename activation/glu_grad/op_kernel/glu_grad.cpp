/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/glu_grad_big_shape.h"
#include "arch35/glu_grad_small_shape.h"
#include "arch35/glu_grad_tiling_key.h"

using namespace GluGrad;

template <typename DTYPE_X, int SHAPE_MODE>
__global__ __aicore__ void glu_grad(GM_ADDR grad_out, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if constexpr (SHAPE_MODE == TPL_SMALL_SHAPE) {
        GluGradSmallShape<DTYPE_X> op;
        op.Init(grad_out, self, out, userWS, &tilingData);
        op.Process();
    } else if constexpr (SHAPE_MODE == TPL_BIG_SHAPE) {
        GluGradBigShape<DTYPE_X> op;
        op.Init(grad_out, self, out, userWS, &tilingData);
        op.Process();
    }
}
