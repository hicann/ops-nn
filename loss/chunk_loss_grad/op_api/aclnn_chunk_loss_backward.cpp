/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "aclnn_chunk_loss_backward.h"
#include "chunk_loss_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include <limits>
#include <cmath>
using namespace op;
namespace {
struct Inputs {
    const aclTensor *hidden, *weight, *labels, *softmaxMax, *softmaxSum, *gradOutput;
};
bool HasShape(const aclTensor* tensor, const Shape& shape) { return tensor->GetViewShape() == shape; }
bool CheckShape(const Inputs& inputs, const aclTensor* gradWeightOut)
{
    if (inputs.hidden->GetViewShape().GetDimNum() != 2 || inputs.weight->GetViewShape().GetDimNum() != 2) {
        return false;
    }
    int64_t numTokens = inputs.hidden->GetViewShape()[0], hiddenDim = inputs.hidden->GetViewShape()[1];
    int64_t vocabSize = inputs.weight->GetViewShape()[0];
    if (numTokens <= 0 || hiddenDim <= 0 || vocabSize <= 0 || numTokens > INT32_MAX || hiddenDim > INT32_MAX ||
        vocabSize > INT32_MAX) {
        return false;
    }
    // Guard tensor bytes and the int64 element offsets used by the kernel.
    constexpr int64_t maxTensorElements = std::numeric_limits<int64_t>::max() / sizeof(float);
    if (numTokens > maxTensorElements / hiddenDim || vocabSize > maxTensorElements / hiddenDim) {
        return false;
    }
    return HasShape(inputs.weight, {vocabSize, hiddenDim}) && HasShape(inputs.labels, {numTokens}) &&
           HasShape(inputs.softmaxMax, {numTokens}) && HasShape(inputs.softmaxSum, {numTokens}) &&
           HasShape(inputs.gradOutput, {1}) && HasShape(gradWeightOut, {vocabSize, hiddenDim});
}
aclnnStatus CheckParams(const Inputs& inputs, double alpha, aclTensor* gradWeightOut)
{
    for (auto tensor : {inputs.hidden, inputs.weight, inputs.labels, inputs.softmaxMax, inputs.softmaxSum,
                        inputs.gradOutput, static_cast<const aclTensor*>(gradWeightOut)}) {
        OP_CHECK_NULL(tensor, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK(tensor->GetStorageFormat() == Format::FORMAT_ND,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkLossBackward requires ND tensors."),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    for (auto tensor :
         {inputs.hidden, inputs.weight, inputs.labels, inputs.softmaxMax, inputs.softmaxSum, inputs.gradOutput}) {
        OP_CHECK(tensor != gradWeightOut, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output must not alias inputs."),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    auto hiddenDtype = inputs.hidden->GetDataType();
    OP_CHECK((hiddenDtype == DataType::DT_FLOAT16 || hiddenDtype == DataType::DT_BF16) &&
                 inputs.weight->GetDataType() == hiddenDtype && gradWeightOut->GetDataType() == hiddenDtype &&
                 inputs.labels->GetDataType() == DataType::DT_INT64 &&
                 inputs.softmaxMax->GetDataType() == DataType::DT_FLOAT &&
                 inputs.softmaxSum->GetDataType() == DataType::DT_FLOAT &&
                 inputs.gradOutput->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkLossBackward dtype combination is invalid."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(CheckShape(inputs, gradWeightOut), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkLossBackward shape mismatch."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(std::isfinite(alpha) && alpha > 0.0 && alpha <= std::numeric_limits<float>::max(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha must be positive, finite and representable in FLOAT32."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(static_cast<float>(alpha) > 0.0f,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha must not round to zero in FLOAT32."),
             return ACLNN_ERR_PARAM_INVALID);
    auto soc = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(soc == SocVersion::ASCEND910B || soc == SocVersion::ASCEND910_93 || soc == SocVersion::ASCEND950,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkLossBackward requires A2, A3 or A5."),
             return ACLNN_ERR_PARAM_INVALID);
    // Device values (labels and caches) are caller preconditions: no D2H here.
    return ACLNN_SUCCESS;
}

} // namespace
extern "C" aclnnStatus aclnnChunkLossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* hidden,
                                                              const aclTensor* weight, const aclTensor* labels,
                                                              const aclTensor* softmaxMax, const aclTensor* softmaxSum,
                                                              double alpha, int64_t ignoreIndex,
                                                              aclTensor* gradWeightOut, uint64_t* workspaceSize,
                                                              aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnChunkLossBackward,
                   DFX_IN(gradOutput, hidden, weight, labels, softmaxMax, softmaxSum, alpha, ignoreIndex),
                   DFX_OUT(gradWeightOut));
    Inputs inputs{hidden, weight, labels, softmaxMax, softmaxSum, gradOutput};
    auto status = CheckParams(inputs, alpha, gradWeightOut);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    auto executorPtr = CREATE_EXECUTOR();
    CHECK_RET(executorPtr.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // Materialize strided inputs and canonicalize their view/storage metadata.
    // The native kernel always receives dense tensors; ViewCopy below restores
    // the caller-provided output strides and storage offset.
    Inputs contiguous{l0op::Contiguous(hidden, executorPtr.get()),     l0op::Contiguous(weight, executorPtr.get()),
                      l0op::Contiguous(labels, executorPtr.get()),     l0op::Contiguous(softmaxMax, executorPtr.get()),
                      l0op::Contiguous(softmaxSum, executorPtr.get()), l0op::Contiguous(gradOutput, executorPtr.get())};
    CHECK_RET(contiguous.hidden && contiguous.weight && contiguous.labels && contiguous.softmaxMax &&
                  contiguous.softmaxSum && contiguous.gradOutput,
              ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::ChunkLossGrad(contiguous.hidden, contiguous.weight, contiguous.labels,
                                      static_cast<float>(alpha), contiguous.softmaxMax, contiguous.softmaxSum,
                                      contiguous.gradOutput, ignoreIndex, executorPtr.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, gradWeightOut, executorPtr.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = executorPtr->GetWorkspaceSize();
    executorPtr.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnChunkLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkLossBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
