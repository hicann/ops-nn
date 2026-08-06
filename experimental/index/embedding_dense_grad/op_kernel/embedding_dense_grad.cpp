/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad.cpp
 * \brief kernel entry of embedding_dense_grad
 */
#include "embedding_dense_grad.h"
#include "embedding_dense_grad_scale.h"
#include "embedding_dense_grad_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void embedding_dense_grad(GM_ADDR grad, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    if constexpr (schMode == EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW) {
        AscendC::EmbeddingDenseGradKernel<DTYPE_GRAD, DTYPE_INDICES, EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW> op(
            grad, indices, y, workspace, tiling_data, pipe);
        op.Process();
        if (tiling_data.scaleGradByFreq) {
            pipe.Destroy();
            AscendC::TPipe pipe2;
            AscendC::EmbeddingDenseGradScaleKernel<DTYPE_GRAD> op2(y, workspace, tiling_data, pipe2);
            op2.Process();
        } else {
            op.CastOutput();
        }
    } else if constexpr (schMode == EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED) {
        AscendC::EmbeddingDenseGradKernel<DTYPE_GRAD, DTYPE_INDICES, EMBEDDING_DENSE_GRAD_SCH_MODE_SEGMENTED> op(
            grad, indices, y, workspace, tiling_data, pipe);
        op.Process();
        if (tiling_data.scaleGradByFreq) {
            pipe.Destroy();
            AscendC::TPipe pipe2;
            AscendC::EmbeddingDenseGradScaleKernel<DTYPE_GRAD> op2(y, workspace, tiling_data, pipe2);
            op2.Process();
        } else {
            op.CastOutput();
        }
    } else if constexpr (schMode == EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED) {
        AscendC::EmbeddingDenseGradKernel<DTYPE_GRAD, DTYPE_INDICES, EMBEDDING_DENSE_GRAD_SCH_MODE_PACKED> op(
            grad, indices, y, workspace, tiling_data, pipe);
        op.Process();
    }
}
