/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad.cpp
 * \brief embedding_dense_grad
 */

#include "./arch35/embedding_dense_grad_base.h"
#include "./arch35/embedding_dense_grad_full_load.h"
#include "./arch35/embedding_dense_grad.h"
using namespace AscendC;
using namespace EmbeddingDenseGrad;

#define EMBEDDING_BF16_INT32 401
#define EMBEDDING_FP16_INT32 402
#define EMBEDDING_FP32_INT32 404
#define EMBEDDING_BF16_INT64 801
#define EMBEDDING_FP16_INT64 802
#define EMBEDDING_FP32_INT64 804

#define EMBEDDING_FULL_IDX_BF16_NO_SCALE 101
#define EMBEDDING_FULL_IDX_FP16_NO_SCALE 102
#define EMBEDDING_FULL_IDX_FP32_NO_SCALE 104
#define EMBEDDING_FULL_IDX_BF16_SCALE 111
#define EMBEDDING_FULL_IDX_FP16_SCALE 112
#define EMBEDDING_FULL_IDX_FP32_SCALE 114

extern "C" __global__ __aicore__ void embedding_dense_grad(GM_ADDR grad, GM_ADDR indices, GM_ADDR gradWeight,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe tpipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(EMBEDDING_BF16_INT32)) {
        KernelEmbeddingDenseGrad<bfloat16_t, int32_t, uint16_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FP16_INT32)) {
        KernelEmbeddingDenseGrad<half, int32_t, uint16_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FP32_INT32)) {
        KernelEmbeddingDenseGrad<float, int32_t, uint32_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_BF16_INT64)) {
        KernelEmbeddingDenseGrad<bfloat16_t, int64_t, uint16_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FP16_INT64)) {
        KernelEmbeddingDenseGrad<half, int64_t, uint16_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FP32_INT64)) {
        KernelEmbeddingDenseGrad<float, int64_t, uint32_t> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_BF16_NO_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<bfloat16_t, DTYPE_INDICES, uint16_t, false> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_FP16_NO_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<half, DTYPE_INDICES, uint16_t, false> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_FP32_NO_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<float, DTYPE_INDICES, uint32_t, false> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_BF16_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<bfloat16_t, DTYPE_INDICES, uint16_t, true> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_FP16_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<half, DTYPE_INDICES, uint16_t, true> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    } else if (TILING_KEY_IS(EMBEDDING_FULL_IDX_FP32_SCALE)) {
        KernelEmbeddingDenseGradFullLoad<float, DTYPE_INDICES, uint32_t, true> op(tpipe, tilingData);
        op.Init(grad, indices, gradWeight, userWS);
        op.Process();
    }
}