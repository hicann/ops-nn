/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nll_loss.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NLLLoss);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND91095_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static inline const std::initializer_list<op::DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND910_95: {
            return ASCEND91095_AICORE_DTYPE_SUPPORT_LIST;
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
    return CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

// AICORE算子kernel
static std::array<const aclTensor*, 2> NLLLossAiCore(
    const aclTensor* x, const aclTensor* target, const aclTensor* weight, const std::string& reduction,
    int64_t ignoreIndex, const aclTensor* out, const aclTensor* targetWeightOut, aclOpExecutor* executor)
{
    L0_DFX(NLLLossAiCore, x, target, weight, out, targetWeightOut);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        NLLLoss, OP_INPUT(x, target, weight), OP_OUTPUT(out, targetWeightOut), OP_ATTR(reduction, ignoreIndex));
    if (retAicore != ACLNN_SUCCESS) {
        return {nullptr, nullptr};
    }
    return {out, targetWeightOut};
}

// AICPU算子kernel
static std::array<const aclTensor*, 2> NLLLossAiCpu(
    const aclTensor* x, const aclTensor* target, const aclTensor* weight, const std::string& reduction,
    int64_t ignoreIndex, const aclTensor* out, const aclTensor* targetWeightOut, aclOpExecutor* executor)
{
    L0_DFX(NLLLossAiCpu, x, target, weight, out, targetWeightOut);
    static internal::AicpuTaskSpace space("NLLLoss");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        NLLLoss, OP_ATTR_NAMES({"reduction", "ignore_index"}), OP_INPUT(x, target, weight),
        OP_OUTPUT(out, targetWeightOut), OP_ATTR(reduction, ignoreIndex));
    if (ret != ACLNN_SUCCESS) {
        return {nullptr, nullptr};
    }
    return {out, targetWeightOut};
}

const std::array<const aclTensor*, 2> NLLLoss(
    const aclTensor* x, const aclTensor* target, const aclTensor* weight, const std::string& reduction,
    int64_t ignoreIndex, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto totalWeightOut = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        NLLLoss, OP_INPUT(x, target, weight), OP_OUTPUT(out, totalWeightOut), OP_ATTR(reduction.c_str(), ignoreIndex));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return {};
    }
    if (IsAiCoreSupport(x)) {
        return NLLLossAiCore(x, target, weight, reduction, ignoreIndex, out, totalWeightOut, executor);
    } else {
        return NLLLossAiCpu(x, target, weight, reduction, ignoreIndex, out, totalWeightOut, executor);
    }
}

} // namespace l0op
