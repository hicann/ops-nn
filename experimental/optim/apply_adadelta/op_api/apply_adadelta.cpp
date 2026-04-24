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
 * @file apply_adadelta.cpp
 * @brief ACLNN L0 API implementation for ApplyAdadelta
 *
 * L0 API: InferShape, IsAiCoreSupport, AllocTensor, ApplyAdadeltaAiCore
 *
 * This is an inplace operator. The Kernel receives both input and output GM addresses,
 * where output addresses overlap with input addresses (var/accum/accumUpdate).
 */

#include "apply_adadelta.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyAdadelta);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16
};

static bool IsAiCoreSupport(const aclTensor* var)
{
    return CheckType(var->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* ApplyAdadeltaAiCore(
    const aclTensor* var,
    const aclTensor* accum,
    const aclTensor* accumUpdate,
    const aclTensor* grad,
    const aclTensor* varOut,
    const aclTensor* accumOut,
    const aclTensor* accumUpdateOut,
    float lr, float rho, float epsilon,
    aclOpExecutor* executor)
{
    L0_DFX(ApplyAdadeltaAiCore, var, accum, accumUpdate, grad, varOut, accumOut, accumUpdateOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdadelta,
        OP_INPUT(var, accum, accumUpdate, grad),
        OP_OUTPUT(varOut, accumOut, accumUpdateOut),
        OP_ATTR(lr, rho, epsilon));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyAdadeltaAiCore failed."),
        return nullptr);
    return varOut;
}

ApplyAdadeltaOutputs ApplyAdadelta(
    const aclTensor* var,
    const aclTensor* accum,
    const aclTensor* accumUpdate,
    const aclTensor* grad,
    float lr,
    float rho,
    float epsilon,
    aclOpExecutor* executor)
{
    ApplyAdadeltaOutputs emptyResult = {nullptr, nullptr, nullptr};

    if (!IsAiCoreSupport(var)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ApplyAdadelta not supported: dtype=%d.",
                static_cast<int>(var->GetDataType()));
        return emptyResult;
    }

    // Output shape = input shape (inplace)
    auto varShape = var->GetViewShape();

    // Allocate output tensors (inplace: framework handles address aliasing;
    // the L2 API completes aliasing / write-back to the original var/accum/
    // accumUpdate buffers via ViewCopy after this L0 call returns.)
    const aclTensor* varOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* accumOut = executor->AllocTensor(varShape, var->GetDataType());
    const aclTensor* accumUpdateOut = executor->AllocTensor(varShape, var->GetDataType());

    const aclTensor* result = ApplyAdadeltaAiCore(var, accum, accumUpdate, grad,
                                varOut, accumOut, accumUpdateOut,
                                lr, rho, epsilon, executor);
    if (result == nullptr) {
        return emptyResult;
    }

    return {varOut, accumOut, accumUpdateOut};
}

} // namespace l0op
