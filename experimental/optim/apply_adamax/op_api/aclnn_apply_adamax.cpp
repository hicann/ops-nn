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
 * @file aclnn_apply_adamax.cpp
 * @brief ACLNN L2 API implementation for ApplyAdamax
 *
 * Two-stage interface:
 * 1. aclnnApplyAdamaxGetWorkspaceSize - parameter checking, Contiguous, L0 dispatch
 * 2. aclnnApplyAdamax - execute computation
 *
 * Scalar parameters (beta1Power, lr, beta1, beta2, epsilon) are extracted from
 * aclScalar via ToFloat() and passed as attrs to the Tiling function via the
 * executor (refactored from the original aclTensor[1] design).
 */

#include "aclnn_apply_adamax.h"
#include "apply_adamax.h"
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

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16
};

static bool CheckNotNull(
    const aclTensor* var, const aclTensor* m, const aclTensor* v,
    const aclScalar* beta1Power, const aclScalar* lr,
    const aclScalar* beta1, const aclScalar* beta2, const aclScalar* epsilon,
    const aclTensor* grad, const aclTensor* varOut,
    const aclTensor* mOut, const aclTensor* vOut)
{
    OP_CHECK_NULL(var, return false);
    OP_CHECK_NULL(m, return false);
    OP_CHECK_NULL(v, return false);
    OP_CHECK_NULL(beta1Power, return false);
    OP_CHECK_NULL(lr, return false);
    OP_CHECK_NULL(beta1, return false);
    OP_CHECK_NULL(beta2, return false);
    OP_CHECK_NULL(epsilon, return false);
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(varOut, return false);
    OP_CHECK_NULL(mOut, return false);
    OP_CHECK_NULL(vOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* var, const aclTensor* m,
    const aclTensor* v, const aclTensor* grad)
{
    auto dtype = var->GetDataType();
    if (!CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported dtype: %d. Only FLOAT and FLOAT16 are supported.",
                static_cast<int>(dtype));
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(m, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(v, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(grad, dtype, return false);
    return true;
}

static bool CheckShapeConsistent(
    const aclTensor* var, const aclTensor* m,
    const aclTensor* v, const aclTensor* grad)
{
    OP_CHECK_MAX_DIM(var, ACLNN_MAX_SHAPE_RANK, return false);

    auto varShape = var->GetViewShape();
    auto mShape = m->GetViewShape();
    auto vShape = v->GetViewShape();
    auto gradShape = grad->GetViewShape();

    if (varShape.GetDimNum() == 0 || mShape.GetDimNum() == 0 ||
        vShape.GetDimNum() == 0 || gradShape.GetDimNum() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "0-dim scalar Tensor is not supported; supported dim range is 1~%d",
                ACLNN_MAX_SHAPE_RANK);
        return false;
    }

    if (varShape.GetDimNum() != mShape.GetDimNum() ||
        varShape.GetDimNum() != vShape.GetDimNum() ||
        varShape.GetDimNum() != gradShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Shape dims mismatch: var=%zu, m=%zu, v=%zu, grad=%zu",
                varShape.GetDimNum(), mShape.GetDimNum(),
                vShape.GetDimNum(), gradShape.GetDimNum());
        return false;
    }

    for (size_t i = 0; i < varShape.GetDimNum(); i++) {
        if (varShape.GetDim(i) != mShape.GetDim(i) ||
            varShape.GetDim(i) != vShape.GetDim(i) ||
            varShape.GetDim(i) != gradShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Shape mismatch at dim %zu: var=%ld, m=%ld, v=%ld, grad=%ld",
                    i, varShape.GetDim(i), mShape.GetDim(i),
                    vShape.GetDim(i), gradShape.GetDim(i));
            return false;
        }
    }
    return true;
}

// Validate scalar values (per README declared ranges):
//   - epsilon must be >= 0 (Kernel-level floor guard handles == 0)
//   - beta1Power must be in [0, 1)  (avoid 1 - beta1Power == 0)
//   - beta1 must be in [0, 1)
//   - beta2 must be in [0, 1)
static bool CheckScalarValues(
    const aclScalar* beta1Power, const aclScalar* lr,
    const aclScalar* beta1, const aclScalar* beta2, const aclScalar* epsilon)
{
    float b1pVal = beta1Power->ToFloat();
    float epsVal = epsilon->ToFloat();
    float b1Val = beta1->ToFloat();
    float b2Val = beta2->ToFloat();
    (void)lr;
    if (epsVal < 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "epsilon must be >= 0, got %f", epsVal);
        return false;
    }
    if (b1pVal < 0.0f || b1pVal >= 1.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "beta1Power must be in [0, 1), got %f", b1pVal);
        return false;
    }
    if (b1Val < 0.0f || b1Val >= 1.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "beta1 must be in [0, 1), got %f", b1Val);
        return false;
    }
    if (b2Val < 0.0f || b2Val >= 1.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "beta2 must be in [0, 1), got %f", b2Val);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* var, const aclTensor* m, const aclTensor* v,
    const aclScalar* beta1Power, const aclScalar* lr,
    const aclScalar* beta1, const aclScalar* beta2, const aclScalar* epsilon,
    const aclTensor* grad, const aclTensor* varOut,
    const aclTensor* mOut, const aclTensor* vOut)
{
    if (!CheckNotNull(var, m, v, beta1Power, lr, beta1, beta2, epsilon, grad,
                      varOut, mOut, vOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Null pointer in input parameters");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(var, m, v, grad)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShapeConsistent(var, m, v, grad)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckScalarValues(beta1Power, lr, beta1, beta2, epsilon)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyAdamaxGetWorkspaceSize(
    const aclTensor* var,
    const aclTensor* m,
    const aclTensor* v,
    const aclScalar* beta1Power,
    const aclScalar* lr,
    const aclScalar* beta1,
    const aclScalar* beta2,
    const aclScalar* epsilon,
    const aclTensor* grad,
    aclTensor* varOut,
    aclTensor* mOut,
    aclTensor* vOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnApplyAdamax,
                    DFX_IN(var, m, v, grad),
                    DFX_OUT(varOut, mOut, vOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(var, m, v, beta1Power, lr, beta1, beta2, epsilon, grad,
                           varOut, mOut, vOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (var->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make inputs contiguous
    auto varContig = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mContig = l0op::Contiguous(m, uniqueExecutor.get());
    CHECK_RET(mContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vContig = l0op::Contiguous(v, uniqueExecutor.get());
    CHECK_RET(vContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContig = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Extract scalar values and pass to L0 API
    float beta1PowerVal = beta1Power->ToFloat();
    float lrVal = lr->ToFloat();
    float beta1Val = beta1->ToFloat();
    float beta2Val = beta2->ToFloat();
    float epsVal = epsilon->ToFloat();

    auto opResults = l0op::ApplyAdamax(
        varContig, mContig, vContig, gradContig,
        beta1PowerVal, lrVal, beta1Val, beta2Val, epsVal,
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

extern "C" aclnnStatus aclnnApplyAdamax(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyAdamax);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
