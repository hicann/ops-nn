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
  *\file fused_linear_cross_entropy_loss_grad.cpp
  *\brief
 */
#include "fused_linear_cross_entropy_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedLinearCrossEntropyLossGrad);
static std::tuple<aclTensor *, aclTensor *> errorRes = {nullptr, nullptr};

static std::tuple<aclTensor *, aclTensor *> GetOutTensor(
    const aclTensor *grad, const aclTensor *weight, aclOpExecutor *executor)
{
    // 获取shape
    int64_t BT = grad->GetViewShape().GetDim(0);
    int64_t V = weight->GetViewShape().GetDim(0);
    int64_t H = weight->GetViewShape().GetDim(1);
    // 推断shape
    op::Shape gradInputShape;
    gradInputShape.AppendDim(BT);
    gradInputShape.AppendDim(H);
    op::Shape gradWeightShape;
    gradWeightShape.AppendDim(V);
    gradWeightShape.AppendDim(H);
    // 创建输出张量
    auto gradInput = executor->AllocTensor(
        gradInputShape, gradInputShape, weight->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND
    );
    auto gradWeight = executor->AllocTensor(
        gradWeightShape, gradWeightShape, weight->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND
    );
    return {gradInput, gradWeight};
}

const std::tuple<aclTensor *, aclTensor *> FusedLinearCrossEntropyLossGrad(
    const aclTensor *grad, const aclTensor *input, const aclTensor *weight,
    const aclTensor *targetMask, const aclTensor *maskedTarget, const aclTensor *logitsMax, const aclTensor *sumExpLogits,
    const aclTensor *softmax, const float labelSmoothing, aclOpExecutor *executor)
{
    std::tuple<aclTensor *, aclTensor *> out = GetOutTensor(grad, weight, executor);
    L0_DFX(
        FusedLinearCrossEntropyLossGrad,
        grad, input, weight, targetMask, maskedTarget, logitsMax, sumExpLogits, softmax,
        std::get<0>(out), std::get<1>(out),
        labelSmoothing
    );

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FusedLinearCrossEntropyLossGrad,
        OP_INPUT(grad, input, weight, targetMask, maskedTarget, logitsMax, sumExpLogits, softmax),
        OP_OUTPUT(std::get<0>(out), std::get<1>(out)),
        OP_ATTR(labelSmoothing)
    );

    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedLinearCrossEntropyLossGrad ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return errorRes
    );

    return out;
}
}  // namespace l0op
