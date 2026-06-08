/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gelu_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "gelu_grad.h"
#include "aclnn_kernels/cast.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base.h"
#include "op_api/aclnn_util.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

// self、gradInput的shape需要满足broadcast规则，gradnput的shape为broadcast后的shape
static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    const auto& gradInputShape = gradInput->GetViewShape();

    op::Shape shapeBroadcast;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, shapeBroadcast, return false);

    if (shapeBroadcast != gradInputShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(shapeBroadcast).GetString(), op::ToString(gradInputShape).GetString());
        return false;
    }
    return true;
}

namespace {
static bool CheckPromoteType(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* /* gradInput */)
{
    op::DataType promoteType;
    if (!CheckPromoteTypeGeluBackward(gradOutput, self, promoteType, DTYPE_SUPPORT_LIST)) {
        return false;
    }
    return true;
}
} // namespace

// getBroadcastShape for l0op::BroadcastTo
static aclIntArray* GetShape(const op::Shape broadcastShape, aclOpExecutor* executor)
{
    int64_t tensorSize = (int64_t)(broadcastShape.GetDimNum());
    std::vector<int64_t> tensorShape(tensorSize);
    for (int i = 0; i < tensorSize; i++) {
        tensorShape[i] = broadcastShape[i];
    }
    return executor->AllocIntArray(tensorShape.data(), tensorSize);
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查gradOutput和self的dtype能否作类型推导,类型推导后的结果需要与gradInput一致
    CHECK_RET(CheckPromoteType(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查gradOutput和self以及gradInput的维度匹配关系
    CHECK_RET(CheckShape(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 如果gradOutput或者self的shape与broadcast后的shape不一致，在进行反向计算前，先进行broadcasto操作。
static const aclTensor* BroadcastTensor(const aclTensor* self, const op::Shape broadcastShape, aclOpExecutor* executor)
{
    // 如果self的shape与broadcast的不一致，进行BroadcastTo
    if (self->GetViewShape() != broadcastShape) {
        auto broadcastShapeIntArray = GetShape(broadcastShape, executor);
        if (broadcastShapeIntArray != nullptr) {
            return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
        }
    }
    return self;
}

static aclnnStatus RunGeluGradAndCopy(
    const aclTensor* gradOutputCasted, const aclTensor* selfCasted, const aclTensor* gradInputCasted,
    const aclTensor* gradInput, aclOpExecutor* executor)
{
    auto GeluBackwardResult =
        l0op::GeluGrad(gradOutputCasted, selfCasted, gradOutputCasted, gradInputCasted, executor);
    CHECK_RET(GeluBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(GeluBackwardResult, gradInput->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, gradInput, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeGeluGradRegbase(
    const aclTensor* gradOutputContiguous, const aclTensor* selfContiguous, const aclTensor* gradInputCasted,
    const aclTensor* gradInput, op::DataType promoteType, aclOpExecutor* executor)
{
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor);
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, executor);
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return RunGeluGradAndCopy(gradOutputCasted, selfCasted, gradInputCasted, gradInput, executor);
}

static aclnnStatus ComputeGeluGradBroadcast(
    const aclTensor* gradOutputContiguous, const aclTensor* selfContiguous, const aclTensor* gradInputCasted,
    const aclTensor* gradInput, op::DataType promoteType, const op::Shape& broadcastShape, aclOpExecutor* executor)
{
    auto gradOutputBroadcast = BroadcastTensor(gradOutputContiguous, broadcastShape, executor);
    CHECK_RET(gradOutputBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfBroadcast = BroadcastTensor(selfContiguous, broadcastShape, executor);
    CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfBroadcast, promoteType, executor);
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputCasted = l0op::Cast(gradOutputBroadcast, promoteType, executor);
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return RunGeluGradAndCopy(gradOutputCasted, selfCasted, gradInputCasted, gradInput, executor);
}

aclnnStatus aclnnGeluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeluBackward, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradOutput, self, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    op::Shape broadcastShape;
    BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape);
    auto promoteType = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
    CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputCasted = l0op::Cast(gradInputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradInputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclnnStatus status;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        status = ComputeGeluGradRegbase(gradOutputContiguous, selfContiguous, gradInputCasted,
                                        gradInput, promoteType, uniqueExecutor.get());
    } else {
        status = ComputeGeluGradBroadcast(gradOutputContiguous, selfContiguous, gradInputCasted,
                                          gradInput, promoteType, broadcastShape, uniqueExecutor.get());
    }
    CHECK_RET(status == ACLNN_SUCCESS, status);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluBackward(
    void* workspace, uint64_t workspace_size, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeluBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}