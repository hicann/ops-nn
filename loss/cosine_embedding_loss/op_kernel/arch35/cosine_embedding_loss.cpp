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
 * \file cosine_embedding_loss.cpp
 * \brief CosineEmbeddingLoss arch35 (ascend950) kernel entry.
 *
 * Input dtypes are provided by the generated DTYPE_X1/DTYPE_X2/DTYPE_TARGET compile options,
 * so every binary instantiates only its configured types.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cosine_embedding_loss.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void cosine_embedding_loss(GM_ADDR x1, GM_ADDR x2, GM_ADDR target, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace); // per-core partial area
    REGISTER_TILING_DEFAULT(CosineEmbeddingLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(CosineEmbeddingLossTilingData, tilingData, tiling);
    TPipe pipe;

    NsCosineEmbeddingLoss::CosineEmbeddingLossKernel<DTYPE_X1, DTYPE_X2, DTYPE_TARGET> op;
    op.Init(x1, x2, target, y, userWS, &tilingData, &pipe);
    op.Process();
}
