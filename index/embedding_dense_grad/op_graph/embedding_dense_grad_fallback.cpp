/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
const static size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
const static size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
const static size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;

graphStatus EmbeddingDenseGradHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto inputGrad = host_api_ctx->GetInputTensor(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_IF(inputGrad == nullptr, OP_LOGE("aclnnfallback", "grad is null"), return GRAPH_FAILED);

    auto indices = host_api_ctx->GetInputTensor(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_IF(indices == nullptr, OP_LOGE("aclnnfallback", "indices is null"), return GRAPH_FAILED);

    auto output = host_api_ctx->GetOutputTensor(EMBEDDING_DENSE_GRAD_OUT_Y);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);

    const uint64_t* numWeight = attrs->GetAttrPointer<uint64_t>(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    OP_CHECK_IF(numWeight == nullptr, OP_LOGE("aclnnfallback", "numWeight is null"), return GRAPH_FAILED);
    const uint64_t* paddingIdx = attrs->GetAttrPointer<uint64_t>(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    OP_CHECK_IF(paddingIdx == nullptr, OP_LOGE("aclnnfallback", "paddingIdx is null"), return GRAPH_FAILED);
    const bool* scaleGrad = attrs->GetAttrPointer<bool>(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
    OP_CHECK_IF(scaleGrad == nullptr, OP_LOGE("aclnnfallback", "scaleGrad is null"), return GRAPH_FAILED);

    OP_LOGD("aclnnFallback", "EmbeddingDenseGrad fallback begin");
    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnEmbeddingDenseBackward, inputGrad, indices,
                                               *numWeight, *paddingIdx, *scaleGrad, output);

    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret faild:%d", api_ret), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(EmbeddingDenseGrad).OpExecuteFunc(EmbeddingDenseGradHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
