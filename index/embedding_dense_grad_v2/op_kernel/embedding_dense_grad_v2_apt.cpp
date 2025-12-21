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
 * \file embedding_dense_grad_v2_apt.cpp
 * \brief embedding_dense_grad_v2 kernel
 */

#include "v35/embedding_dense_grad_v2_struct.h"
#include "v35/embedding_dense_grad_v2_regbase.h"
using namespace AscendC;
using namespace EmbeddingDenseGradV2;

#define EMBEDDING_NO_SCALE_NO_SPLIT  1000000
#define EMBEDDING_NO_SCALE_SPLIT  1000001
#define EMBEDDING_SCALE_NO_SPLIT  1000010
#define EMBEDDING_SCALE_SPLIT  1000011

extern "C" __global__ __aicore__ void embedding_dense_grad_v2(GM_ADDR grad, GM_ADDR sortIndices,
    GM_ADDR posIdx, GM_ADDR gradWeight, GM_ADDR workSpace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EmbeddingDenseGradV2TilingData4RegBase);
    GM_ADDR userWS = GetUserWorkspace(workSpace);
    if (userWS == nullptr) {
        return;
    }

    AscendC::TPipe tpipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(EMBEDDING_NO_SCALE_NO_SPLIT)) {
        EmbeddingDenseGradV2::EmbeddingDenseGradV2Kernel<DTYPE_GRAD, float, DTYPE_SORT_INDICES, false, false, 1> obj(tpipe, tilingData);
        obj.Init(grad, sortIndices, posIdx, gradWeight, userWS);
        obj.Process();
    } else if (TILING_KEY_IS(EMBEDDING_NO_SCALE_SPLIT)) {
        EmbeddingDenseGradV2::EmbeddingDenseGradV2Kernel<DTYPE_GRAD, float, DTYPE_SORT_INDICES, false, true, 1> obj(tpipe, tilingData);
        obj.Init(grad, sortIndices, posIdx, gradWeight, userWS);
        obj.Process();
    } else if (TILING_KEY_IS(EMBEDDING_SCALE_NO_SPLIT)) {
        EmbeddingDenseGradV2::EmbeddingDenseGradV2Kernel<DTYPE_GRAD, float, DTYPE_SORT_INDICES, true, false, 1> obj(tpipe, tilingData);
        obj.Init(grad, sortIndices, posIdx, gradWeight, userWS);
        obj.Process();
    } else if (TILING_KEY_IS(EMBEDDING_SCALE_SPLIT)) {
        EmbeddingDenseGradV2::EmbeddingDenseGradV2Kernel<DTYPE_GRAD, float, DTYPE_SORT_INDICES, true, true, 1> obj(tpipe, tilingData);
        obj.Init(grad, sortIndices, posIdx, gradWeight, userWS);
        obj.Process();
    }
}