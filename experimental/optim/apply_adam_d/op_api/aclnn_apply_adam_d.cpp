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
 * @file aclnn_apply_adam_d.cpp
 * @brief ACLNN L2 API implementation for ApplyAdamD.
 */

#include "aclnn_apply_adam_d.h"
#include "apply_adam_d.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckNotNull(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, const aclTensor* varOut, const aclTensor* mOut,
    const aclTensor* vOut, const uint64_t* workspaceSize, aclOpExecutor* const* executor)
{
    OP_CHECK_NULL(var, return false);
    OP_CHECK_NULL(m, return false);
    OP_CHECK_NULL(v, return false);
    OP_CHECK_NULL(beta1Power, return false);
    OP_CHECK_NULL(beta2Power, return false);
    OP_CHECK_NULL(lr, return false);
    OP_CHECK_NULL(beta1, return false);
    OP_CHECK_NULL(beta2, return false);
    OP_CHECK_NULL(epsilon, return false);
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(varOut, return false);
    OP_CHECK_NULL(mOut, return false);
    OP_CHECK_NULL(vOut, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool IsSameShape(const aclTensor* lhs, const aclTensor* rhs)
{
    auto lhsShape = lhs->GetViewShape();
    auto rhsShape = rhs->GetViewShape();
    if (lhsShape.GetDimNum() != rhsShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhsShape.GetDimNum(); ++i) {
        if (lhsShape.GetDim(i) != rhsShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static int64_t GetShapeSize(const aclTensor* tensor)
{
    auto shape = tensor->GetViewShape();
    int64_t size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

static bool CheckDtypeValid(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, const aclTensor* varOut, const aclTensor* mOut,
    const aclTensor* vOut)
{
    auto dtype = var->GetDataType();
    if (!CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Unsupported dtype: %d. Only FLOAT, FLOAT16 and BF16 are supported.",
            static_cast<int>(dtype));
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(m, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(v, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta1Power, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta2Power, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(lr, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta1, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta2, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(epsilon, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(grad, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(varOut, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(mOut, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(vOut, dtype, return false);
    return true;
}

static bool CheckTensorShapes(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* grad, const aclTensor* varOut,
    const aclTensor* mOut, const aclTensor* vOut)
{
    OP_CHECK_MAX_DIM(var, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(m, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(v, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(grad, ACLNN_MAX_SHAPE_RANK, return false);

    if (!IsSameShape(var, m) || !IsSameShape(var, v) || !IsSameShape(var, grad)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "var, m, v and grad must have identical shapes.");
        return false;
    }
    if (!IsSameShape(var, varOut) || !IsSameShape(var, mOut) || !IsSameShape(var, vOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "varOut, mOut and vOut must have the same shape as var.");
        return false;
    }
    return true;
}

static bool CheckScalarTensorShapes(
    const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1,
    const aclTensor* beta2, const aclTensor* epsilon)
{
    const aclTensor* scalars[] = {beta1Power, beta2Power, lr, beta1, beta2, epsilon};
    const char* names[] = {"beta1Power", "beta2Power", "lr", "beta1", "beta2", "epsilon"};
    for (size_t i = 0; i < sizeof(scalars) / sizeof(scalars[0]); ++i) {
        OP_CHECK_MAX_DIM(scalars[i], ACLNN_MAX_SHAPE_RANK, return false);
        if (GetShapeSize(scalars[i]) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be a scalar tensor with shape size 1.", names[i]);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, const aclTensor* varOut, const aclTensor* mOut,
    const aclTensor* vOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (!CheckNotNull(
            var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut, workspaceSize,
            executor)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Null pointer in input parameters.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckTensorShapes(var, m, v, grad, varOut, mOut, vOut)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckScalarTensorShapes(beta1Power, beta2Power, lr, beta1, beta2, epsilon)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyAdamDGetWorkspaceSize(
    const aclTensor* var, const aclTensor* m, const aclTensor* v, const aclTensor* beta1Power,
    const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad, bool useLocking, bool useNesterov, aclTensor* varOut,
    aclTensor* mOut, aclTensor* vOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnApplyAdamD, DFX_IN(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad),
        DFX_OUT(varOut, mOut, vOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(
        var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut, workspaceSize,
        executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (var->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mContiguous = l0op::Contiguous(m, uniqueExecutor.get());
    CHECK_RET(mContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vContiguous = l0op::Contiguous(v, uniqueExecutor.get());
    CHECK_RET(vContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto beta1PowerContiguous = l0op::Contiguous(beta1Power, uniqueExecutor.get());
    CHECK_RET(beta1PowerContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto beta2PowerContiguous = l0op::Contiguous(beta2Power, uniqueExecutor.get());
    CHECK_RET(beta2PowerContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto lrContiguous = l0op::Contiguous(lr, uniqueExecutor.get());
    CHECK_RET(lrContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto beta1Contiguous = l0op::Contiguous(beta1, uniqueExecutor.get());
    CHECK_RET(beta1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto beta2Contiguous = l0op::Contiguous(beta2, uniqueExecutor.get());
    CHECK_RET(beta2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto epsilonContiguous = l0op::Contiguous(epsilon, uniqueExecutor.get());
    CHECK_RET(epsilonContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto opResults = l0op::ApplyAdamD(
        varContiguous, mContiguous, vContiguous, beta1PowerContiguous, beta2PowerContiguous, lrContiguous,
        beta1Contiguous, beta2Contiguous, epsilonContiguous, gradContiguous, useLocking, useNesterov,
        uniqueExecutor.get());
    CHECK_RET(opResults.varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopy1 = l0op::ViewCopy(opResults.varOut, varOut, uniqueExecutor.get());
    CHECK_RET(viewCopy1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopy2 = l0op::ViewCopy(opResults.mOut, mOut, uniqueExecutor.get());
    CHECK_RET(viewCopy2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopy3 = l0op::ViewCopy(opResults.vOut, vOut, uniqueExecutor.get());
    CHECK_RET(viewCopy3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyAdamD(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyAdamD);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
