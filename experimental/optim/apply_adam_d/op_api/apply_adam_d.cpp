/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/**
 * @file apply_adam_d.cpp
 * @brief ACLNN L0 API implementation for ApplyAdamD.
 */

#include "apply_adam_d.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyAdamD);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* var) { return CheckType(var->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

static const aclTensor* ApplyAdamDAiCore(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, const aclTensor* varOut, const aclTensor* mOut,
    const aclTensor* vOut, bool useLocking, bool useNesterov, aclOpExecutor* executor)
{
    L0_DFX(ApplyAdamDAiCore, var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ApplyAdamD, OP_INPUT(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad),
        OP_OUTPUT(varOut, mOut, vOut), OP_ATTR(useLocking, useNesterov));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdamDAiCore failed."), return nullptr);
    return varOut;
}

ApplyAdamDOutputs ApplyAdamD(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, bool useLocking, bool useNesterov, aclOpExecutor* executor)
{
    ApplyAdamDOutputs emptyResult = {nullptr, nullptr, nullptr};

    if (!IsAiCoreSupport(var)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ApplyAdamD not supported: dtype=%d.", static_cast<int>(var->GetDataType()));
        return emptyResult;
    }

    auto varShape = var->GetViewShape();
    const aclTensor* varOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* mOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* vOut = executor->AllocTensor(varShape, var->GetDataType());
    if (varOut == nullptr || mOut == nullptr || vOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdamD output tensor allocation failed.");
        return emptyResult;
    }

    const aclTensor* result = ApplyAdamDAiCore(
        var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut, useLocking, useNesterov,
        executor);
    if (result == nullptr) {
        return emptyResult;
    }

    return {varOut, mOut, vOut};
}

} // namespace l0op
