/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_adaptive_max_pool3d_backward.h"
#include "adaptive_max_pool3d_backward.h"
#include "level0/max_pool3d_grad_with_argmax.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/framework_op.h"
#include "pooling/pool_3d_common/op_api/pool_3d_util.h"
#include "adaptive_max_pool_common.h"
using namespace op;
using namespace AdaptiveMaxPoolCommon;
using namespace Pool3DCommon;
#ifdef __cplusplus
extern "C" {
#endif
static const size_t CDHW_DIMS = 4;
static const size_t NCDHW_DIMS = 5;

static const int W_DIM = -1;
static const int H_DIM = -2;
static const int D_DIM = -3;

static const int64_t MAX_INT32 = 2147483647;
static const std::initializer_list<op::DataType> POOL3D_INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};

static bool CheckFormat(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    if ((self->GetStorageFormat() != gradOutput->GetStorageFormat()) ||
        (self->GetStorageFormat() != gradInput->GetStorageFormat()) ||
        (self->GetStorageFormat() != indices->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of self and gradOutput and gradInput and indices should be same, self[%s], gradOutput[%s], "
            "gradInput[%s], indices[%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString(),
            op::ToString(indices->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(gradInput->GetStorageFormat()) ||
        op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(indices->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCDHW or CDHW");
        return false;
    }

    if (self->GetViewShape().GetDimNum() != CDHW_DIMS && self->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for self but got dim num:%zu",
            self->GetViewShape().GetDimNum());
        return false;
    }

    if (gradInput->GetViewShape().GetDimNum() != CDHW_DIMS && gradInput->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for gradInput but got dim num:%zu",
            gradInput->GetViewShape().GetDimNum());
        return false;
    }

    if (gradOutput->GetViewShape().GetDimNum() != CDHW_DIMS && gradOutput->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for gradOutput but got dim num:%zu",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }

    if (indices->GetViewShape().GetDimNum() != CDHW_DIMS && indices->GetViewShape().GetDimNum() != NCDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for indices but got dim num:%zu",
            indices->GetViewShape().GetDimNum());
        return false;
    }

    return true;
}

static bool CheckSelfShapeSupport(const aclTensor* self)
{   
    const auto& selfShape = self->GetViewShape();
    const auto& selfDimNum = selfShape.GetDimNum();
    const auto& selfDimD = selfShape.GetDim(selfDimNum + D_DIM);
    const auto& selfDimW = selfShape.GetDim(selfDimNum + W_DIM);
    const auto& selfDimH = selfShape.GetDim(selfDimNum + H_DIM);
    

    const int64_t selfSize = selfDimW * selfDimH * selfDimD;
    OP_CHECK(
        (selfSize <= MAX_INT32),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The size of self should be less than or equal to 2^32 - 1, but got selfSize:%ld",
            selfSize),
        return false);
    return true;
}

aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaptiveMaxPool3dBackward, DFX_IN(gradOutput, self, indices), DFX_OUT(gradInput));
    OP_LOGD("AdaptiveMaxPool3DGrad: getWorkspaceSize");

    // Create an OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Сhecking parameters
    auto ret = CheckParams(gradOutput, self, indices, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckSelfShapeSupport(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, POOL3D_INDICES_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    // Check whether the tensor is empty (the operator does not support empty tensors)
    if (gradOutput->IsEmpty() || self->IsEmpty() || indices->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Convert the self, gradOutput, indices into consecutive tensor

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If it's 4D, it needs to be expanded to 5D before calling the AdaptiveMaxPool3DGrad interface.
    const bool isSelf4D = self->GetViewShape().GetDimNum() == CDHW_DIMS;

    auto gradOutputUnsqueezed =
        isSelf4D ? ViewCDHWas5D(gradOutputContiguous, uniqueExecutor.get()) :
                   l0op::ReFormat(gradOutputContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfUnsqueezed = isSelf4D ? ViewCDHWas5D(selfContiguous, uniqueExecutor.get()) :
                                     l0op::ReFormat(selfContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesUnsqueezed = isSelf4D ?
                                 ViewCDHWas5D(indicesContiguous, uniqueExecutor.get()) :
                                 l0op::ReFormat(indicesContiguous, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputResult = selectLevelZeroOperation(
        gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = gradInput->GetStorageFormat();
    auto gradResultSqueezed = isSelf4D ? View5DasCDHW(gradInputResult, dstFormat, uniqueExecutor.get()) :
                                         l0op::ReFormat(gradInputResult, dstFormat, uniqueExecutor.get());
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(gradResultSqueezed, gradInput), ACLNN_ERR_PARAM_INVALID);

    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewGradInputCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewGradInputCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveMaxPool3dBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveMaxPool3dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif