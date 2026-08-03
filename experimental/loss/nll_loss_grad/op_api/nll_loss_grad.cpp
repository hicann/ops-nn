/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nll_loss_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NllLossGrad);

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_2201) {
        return false;
    }
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* NLLLossGradAiCore(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                                          const aclTensor* weight, const string& reduction, int64_t ignoreIndex,
                                          const aclTensor* totalWeight, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NLLLossGradAiCore, gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NllLossGrad, OP_INPUT(self, gradOutput, target, weight, totalWeight),
                                           OP_OUTPUT(out), OP_ATTR(reduction, ignoreIndex));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return out;
}

// AICPU算子kernel
static const aclTensor* NLLLossGradAiCpu(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                                         const aclTensor* weight, const string& reduction, int64_t ignoreIndex,
                                         const aclTensor* totalWeight, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NLLLossGradAiCpu, gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight);
    static internal::AicpuTaskSpace space("NllLossGrad");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(NllLossGrad, OP_ATTR_NAMES({"reduction", "ignore_index"}),
                                          OP_INPUT(self, gradOutput, target, weight, totalWeight), OP_OUTPUT(out),
                                          OP_ATTR(reduction, ignoreIndex));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* NLLLossGrad(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
                             const aclTensor* weight, const std::string& reduction, int64_t ignoreIndex,
                             const aclTensor* totalWeight, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetStorageFormat());
    if (IsAiCoreSupport(self)) {
        return NLLLossGradAiCore(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out, executor);
    } else {
        return NLLLossGradAiCpu(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out, executor);
    }
}

} // namespace l0op
