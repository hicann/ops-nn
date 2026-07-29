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
 * @file aclnn_apply_gradient_descent.cpp
 * @brief ACLNN L2 API implementation for ApplyGradientDescent
 *
 * Two-stage interface:
 * 1. aclnnApplyGradientDescentGetWorkspaceSize - parameter checking, Contiguous, L0 dispatch
 * 2. aclnnApplyGradientDescent - execute computation
 *
 * Inplace semantics: var = var - alpha * delta. The kernel writes into a fresh output
 * tensor which is ViewCopy'd back into the caller's var buffer.
 */

#include "aclnn_apply_gradient_descent.h"
#include "apply_gradient_descent.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                                              DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    OP_CHECK_NULL(var, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(delta, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    auto dtype = var->GetDataType();
    if (!CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported dtype: %d. Only BFLOAT16, FLOAT16 and FLOAT are supported.",
                static_cast<int>(dtype));
        return false;
    }
    // var/alpha/delta dtypes must all match.
    OP_CHECK_DTYPE_NOT_MATCH(alpha, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(delta, dtype, return false);
    return true;
}

// Shape rules:
//   - var and delta must have identical shapes (elementwise update).
//   - alpha must be a scalar tensor (exactly 1 element).
//   - var rank must be within [1, 8].
static bool CheckShape(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    OP_CHECK_MAX_DIM(var, ACLNN_MAX_SHAPE_RANK, return false);

    auto varShape = var->GetViewShape();
    auto deltaShape = delta->GetViewShape();

    if (varShape.GetDimNum() != deltaShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "var and delta must have the same shape dims: var=%zu, delta=%zu",
                varShape.GetDimNum(), deltaShape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < varShape.GetDimNum(); i++) {
        if (varShape.GetDim(i) != deltaShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "var and delta shape mismatch at dim %zu: var=%ld, delta=%ld", i,
                    varShape.GetDim(i), deltaShape.GetDim(i));
            return false;
        }
    }

    auto alphaShape = alpha->GetViewShape();
    if (alphaShape.GetShapeSize() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha must be a scalar tensor with exactly 1 element, got %ld elements.",
                alphaShape.GetShapeSize());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    if (!CheckNotNull(var, alpha, delta)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Null pointer in input parameters");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(var, alpha, delta)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(var, alpha, delta)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyGradientDescentGetWorkspaceSize(aclTensor* var, const aclTensor* alpha,
                                                                 const aclTensor* delta, uint64_t* workspaceSize,
                                                                 aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnApplyGradientDescent, DFX_IN(var, alpha, delta), DFX_OUT(var));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(var, alpha, delta);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (var->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make inputs contiguous.
    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_RET(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto deltaContiguous = l0op::Contiguous(delta, uniqueExecutor.get());
    CHECK_RET(deltaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Call L0 API: var_out = var - alpha * delta.
    auto varOut = l0op::ApplyGradientDescent(varContiguous, alphaContiguous, deltaContiguous, uniqueExecutor.get());
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Inplace write-back into the caller's var buffer.
    auto viewCopyResult = l0op::ViewCopy(varOut, var, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyGradientDescent(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyGradientDescent);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
