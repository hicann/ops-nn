/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NLLLossGrad);
OP_TYPE_REGISTER(NLLLossGrad310p);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const int64_t NONE = 0;
static const int64_t MEAN = 1;
static const int64_t SUM = 2;

static inline const std::initializer_list<op::DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND910_95: {
            return ASCEND910_95_AICORE_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND910: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // NLLLossGrad只需要判断dtype
    return CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

// AICORE算子kernel
static const aclTensor* NLLLossGradAiCore(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
    const string& reduction, int64_t ignoreIndex, const aclTensor* totalWeight, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NLLLossGradAiCore, gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight);

    ADD_TO_LAUNCHER_LIST_AICORE(
        NLLLossGrad, OP_INPUT(self, gradOutput, target, weight, totalWeight), OP_OUTPUT(out),
        OP_ATTR(reduction, ignoreIndex));

    return out;
}

static const aclTensor* NLLLossGradAiCore310p(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
    int64_t reduction, int64_t ignoreIndex, const aclTensor* totalWeight, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NLLLossGradAiCore310p, gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight);
    ADD_TO_LAUNCHER_LIST_AICORE(
        NLLLossGrad310p, OP_INPUT(gradOutput, self, target, weight, totalWeight), OP_OUTPUT(out),
        OP_ATTR(reduction, ignoreIndex));

    return out;
}

// AICPU算子kernel
static const aclTensor* NLLLossGradAiCpu(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
    const string& reduction, int64_t ignoreIndex, const aclTensor* totalWeight, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NLLLossGradAiCpu, gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight);
    static internal::AicpuTaskSpace space("NLLLossGrad");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        NLLLossGrad, OP_ATTR_NAMES({"reduction", "ignore_index"}),
        OP_INPUT(self, gradOutput, target, weight, totalWeight), OP_OUTPUT(out), OP_ATTR(reduction, ignoreIndex));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* NLLLossGrad(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight,
    const std::string& reduction, int64_t ignoreIndex, const aclTensor* totalWeight, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetStorageFormat());
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND310P) {
        int64_t reduction_int = 0;
        if (reduction == "none") {
            reduction_int = NONE;
        }
        if (reduction == "mean") {
            reduction_int = MEAN;
        }
        if (reduction == "sum") {
            reduction_int = SUM;
        }
        return NLLLossGradAiCore310p(
            gradOutput, self, target, weight, reduction_int, ignoreIndex, totalWeight, out, executor);
    }
    if (IsAiCoreSupport(self)) {
        return NLLLossGradAiCore(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out, executor);
    } else {
        return NLLLossGradAiCpu(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out, executor);
    }
    return out;
}

} // namespace l0op
