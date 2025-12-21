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

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, broadcastShape, return false);

    if (broadcastShape != gradInputShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(gradInputShape).GetString());
        return false;
    }
    return true;
}

namespace
{
static bool CheckPromoteType(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* /* gradInput */)
{
    op::DataType promoteType;
    if (!CheckPromoteTypeGeluBackward(gradOutput, self, promoteType, DTYPE_SUPPORT_LIST)){
        return false;
    }
    return true;
}
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

// 如果gradOutput或者self的shape与braodcast后的shape不一致，在进行反向计算前，先进行broadcasto操作。
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

aclnnStatus aclnnGeluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeluBackward, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty() || gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    op::Shape broadcastShape;
    BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape);

    // 对gradOutput和self两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());

    // gradOutput如果非连续，需要转连续
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // gradInput如果非连续，需要转连续
    auto gradInputContiguous = l0op::Contiguous(gradInput, uniqueExecutor.get());
    CHECK_RET(gradInputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputCasted = l0op::Cast(gradInputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradInputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910_95 &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入gradOutput的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用l0算子GeluGrad进行计算,输出dtype与gradInput一致
        auto GeluBackwardResult =
            l0op::GeluGrad(gradOutputCasted, selfCasted, gradOutputCasted, gradInputCasted, uniqueExecutor.get());
        CHECK_RET(GeluBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(GeluBackwardResult, gradInput->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(castOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 判断gradOutput是否需要进行broadcast
        auto gradOutputBroadcast = BroadcastTensor(gradOutputContiguous, broadcastShape, uniqueExecutor.get());
        CHECK_RET(gradOutputBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 判断self是否需要进行broadcast
        auto selfBroadcast = BroadcastTensor(selfContiguous, broadcastShape, uniqueExecutor.get());
        CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto selfCasted = l0op::Cast(selfBroadcast, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入gradOutput的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto gradOutputCasted = l0op::Cast(gradOutputBroadcast, promoteType, uniqueExecutor.get());
        CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用l0算子GeluGrad进行计算,输出dtype与gradInput一致
        auto GeluBackwardResult =
            l0op::GeluGrad(gradOutputCasted, selfCasted, gradOutputCasted, gradInputCasted, uniqueExecutor.get());
        CHECK_RET(GeluBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(GeluBackwardResult, gradInput->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(castOut, gradInput, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
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
