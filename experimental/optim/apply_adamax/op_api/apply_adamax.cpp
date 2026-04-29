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
 * @file apply_adamax.cpp
 * @brief ACLNN L0 API implementation for ApplyAdamax
 *
 * The Kernel receives 4 input + 3 output Tensor (inplace via L2 ViewCopy)
 * plus 5 scalar Attrs (beta1Power, lr, beta1, beta2, epsilon).
 */

#include "apply_adamax.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyAdamax);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16
};

static bool IsAiCoreSupport(const aclTensor* var)
{
    return CheckType(var->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* ApplyAdamaxAiCore(
    const aclTensor* var,
    const aclTensor* m,
    const aclTensor* v,
    const aclTensor* grad,
    const aclTensor* varOut,
    const aclTensor* mOut,
    const aclTensor* vOut,
    float beta1Power, float lr, float beta1, float beta2, float epsilon,
    aclOpExecutor* executor)
{
    L0_DFX(ApplyAdamaxAiCore, var, m, v, grad, varOut, mOut, vOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdamax,
        OP_INPUT(var, m, v, grad),
        OP_OUTPUT(varOut, mOut, vOut),
        OP_ATTR(beta1Power, lr, beta1, beta2, epsilon));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdamaxAiCore failed."),
        return nullptr);
    return varOut;
}

ApplyAdamaxOutputs ApplyAdamax(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* grad,
    float beta1Power, float lr, float beta1, float beta2, float epsilon,
    aclOpExecutor* executor)
{
    ApplyAdamaxOutputs emptyResult = {nullptr, nullptr, nullptr};

    if (!IsAiCoreSupport(var)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ApplyAdamax not supported: dtype=%d.",
                static_cast<int>(var->GetDataType()));
        return emptyResult;
    }

    auto varShape = var->GetViewShape();

    const aclTensor* varOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* mOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* vOut = executor->AllocTensor(varShape, var->GetDataType());

    const aclTensor* result = ApplyAdamaxAiCore(var, m, v, grad,
                                varOut, mOut, vOut,
                                beta1Power, lr, beta1, beta2, epsilon, executor);
    if (result == nullptr) {
        return emptyResult;
    }

    return {varOut, mOut, vOut};
}

} // namespace l0op
