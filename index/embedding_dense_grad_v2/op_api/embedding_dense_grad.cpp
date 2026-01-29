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
 * \file embedding_Dense_grad.cpp
 * \brief
 */

#include "embedding_dense_grad.h"
#include "level0/zero_op.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

namespace l0op {

OP_TYPE_REGISTER(EmbeddingDenseGrad);
OP_TYPE_REGISTER(EmbeddingDenseGradV2);

const aclTensor* EmbeddingDenseGradV2(
    const aclTensor* grad, const aclTensor* sortIndices, const aclTensor* posIdx, uint64_t numWeights,
    uint64_t paddingIdx, bool scaleGradByFreq, aclOpExecutor* executor)
{
    L0_DFX(EmbeddingDenseGradV2, grad, sortIndices, posIdx, numWeights, paddingIdx, scaleGradByFreq);
    // shape推导
    auto gradOutputShape = grad->GetViewShape();
    int64_t gradOutputShapeLen = gradOutputShape.GetDimNum();
    int64_t outShapeLen = 2;
    op::Shape outShape;
    outShape.SetDimNum(outShapeLen);
    outShape.SetDim(0, numWeights);
    outShape.SetDim(1, gradOutputShape.GetDim(gradOutputShapeLen - 1));

    // 创建输出Tensor
    auto out = executor->AllocTensor(outShape, grad->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_95) {
        auto outZero = l0op::ZerosLike(out, executor);
        if (outZero == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "exec zero tensor failed.");
            return nullptr;
        }
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            EmbeddingDenseGradV2, OP_INPUT(grad, sortIndices, posIdx), OP_OUTPUT(outZero),
            OP_ATTR(numWeights, paddingIdx, scaleGradByFreq));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingDenseGradV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        return outZero;
    } else {
        auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
            EmbeddingDenseGradV2, OP_INPUT(grad, sortIndices, posIdx), OP_OUTPUT(out),
            OP_ATTR(numWeights, paddingIdx, scaleGradByFreq));
        OP_CHECK(
            ret == ACLNN_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingDenseGradV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
            return nullptr);
        return out;
    }
}

const aclTensor* EmbeddingDenseGrad(
    const aclTensor* grad, const aclTensor* indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
    aclOpExecutor* executor)
{
    L0_DFX(EmbeddingDenseGrad, grad, indices, numWeights, paddingIdx, scaleGradByFreq);
    // shape推导
    auto gradOutputShape = grad->GetViewShape();
    int64_t gradOutputShapeLen = gradOutputShape.GetDimNum();
    int64_t outShapeLen = 2;
    op::Shape outShape;
    outShape.SetDimNum(outShapeLen);
    outShape.SetDim(0, numWeights);
    outShape.SetDim(1, gradOutputShape.GetDim(gradOutputShapeLen - 1));

    // 创建输出Tensor
    auto out = executor->AllocTensor(outShape, grad->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }
    // 调用device的EmbeddingDenseGrad算子
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        EmbeddingDenseGrad, OP_INPUT(grad, indices), OP_OUTPUT(out), OP_ATTR(numWeights, paddingIdx, scaleGradByFreq));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingDenseGradAiCcore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}
} // namespace l0op
