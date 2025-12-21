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
 * \file group_norm.cpp
 * \brief
 */
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "group_norm.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupNorm);

// AICORE算子kernel
static std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNormAICore(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, int64_t numGroups, float eps, bool isTraining,
    aclTensor* y, aclTensor* mean, aclTensor* variance, aclOpExecutor* executor)
{
    L0_DFX(GroupNormAICore, x, gamma, beta, numGroups, eps, isTraining, y, mean, variance);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        GroupNorm, OP_INPUT(x, gamma, beta), OP_OUTPUT(y, mean, variance),
        OP_ATTR(numGroups, op::ToString(op::Format::FORMAT_NHWC).GetString(), eps, isTraining));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr),
        "GroupNorm add to aicore launch list failed.");
    return std::tie(y, mean, variance);
}

const std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNorm(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t numGroups,
    float eps, bool isTraining, aclOpExecutor* executor)
{
    auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    int64_t mvShape = N * numGroups;
    auto mean = executor->AllocTensor(op::Shape({mvShape}), x->GetDataType(), op::Format::FORMAT_ND);
    auto variance = executor->AllocTensor(op::Shape({mvShape}), x->GetDataType(), op::Format::FORMAT_ND);
    return GroupNormAICore(x, gamma, beta, numGroups, eps, isTraining, y, mean, variance, executor);
}

} // namespace l0op
