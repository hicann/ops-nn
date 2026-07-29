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
 * @file apply_gradient_descent.cpp
 * @brief ACLNN L0 API implementation for ApplyGradientDescent
 *
 * L0 API: IsAiCoreSupport, AllocTensor, ApplyGradientDescentAiCore.
 * var_out = var - alpha * delta, with alpha a scalar tensor (1 element).
 */

#include "apply_gradient_descent.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyGradientDescent);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                                              DataType::DT_FLOAT};

static bool IsAiCoreSupport(const aclTensor* var) { return CheckType(var->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

static const aclTensor* ApplyGradientDescentAiCore(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta,
                                                   const aclTensor* varOut, aclOpExecutor* executor)
{
    L0_DFX(ApplyGradientDescentAiCore, var, alpha, delta, varOut);

    // ApplyGradientDescent is the OpType; var/alpha/delta are inputs, varOut is the output.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyGradientDescent, OP_INPUT(var, alpha, delta), OP_OUTPUT(varOut));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyGradientDescentAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return varOut;
}

const aclTensor* ApplyGradientDescent(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta,
                                      aclOpExecutor* executor)
{
    if (!IsAiCoreSupport(var)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ApplyGradientDescent not supported: dtype=%d.",
                static_cast<int>(var->GetDataType()));
        return nullptr;
    }

    // Output shape/dtype match var (elementwise, inplace).
    const aclTensor* varOut = executor->AllocTensor(var->GetViewShape(), var->GetDataType());
    if (varOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc varOut tensor failed.");
        return nullptr;
    }

    return ApplyGradientDescentAiCore(var, alpha, delta, varOut, executor);
}

} // namespace l0op
