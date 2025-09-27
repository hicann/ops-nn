/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "leaky_relu_grad.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(LeakyReluGrad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据芯片类型和数据类型判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor* LeakyReluGradAICpu(
    const aclTensor* gradOutput, const aclTensor* self, const float negativeSlope, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(LeakyReluGradAICpu, gradOutput, self, negativeSlope, out);
    static internal::AicpuTaskSpace space("LeakyReluGrad", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        LeakyReluGrad, OP_ATTR_NAMES({"alpha"}), OP_INPUT(gradOutput, self), OP_OUTPUT(out), OP_ATTR(negativeSlope));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LeakyReluGradAICpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

// AICORE算子kernel
static const aclTensor* LeakyReluGradAICore(
    const aclTensor* gradOutput, const aclTensor* self, const float negativeSlope, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(LeakyReluGradAICore, gradOutput, self, negativeSlope, out);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(LeakyReluGrad, OP_INPUT(gradOutput, self), OP_OUTPUT(out), OP_ATTR(negativeSlope));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LeakyReluGradAICore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* LeakyReluGrad(
    const aclTensor* gradOutput, const aclTensor* self, const float negativeSlope, aclOpExecutor* executor)
{
    Shape broadcastShape;
    if (!BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, self->GetDataType(), self->GetViewFormat());

    if (IsAiCoreSupport(self)) {
        return LeakyReluGradAICore(gradOutput, self, negativeSlope, out, executor);
    }
    return LeakyReluGradAICpu(gradOutput, self, negativeSlope, out, executor);
}
} // namespace l0op