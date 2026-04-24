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
 * @file aclnn_apply_adadelta.cpp
 * @brief ACLNN L2 API implementation for ApplyAdadelta
 *
 * Two-stage interface:
 * 1. aclnnApplyAdadeltaGetWorkspaceSize - parameter checking, Contiguous, L0 dispatch
 * 2. aclnnApplyAdadelta - execute computation
 *
 * Scalar parameters (lr, rho, epsilon) are extracted from aclScalar and passed
 * as attrs to the Tiling function via the executor.
 */

#include "aclnn_apply_adadelta.h"
#include "apply_adadelta.h"
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
    const aclTensor* var, const aclTensor* accum, const aclTensor* accumUpdate,
    const aclScalar* lr, const aclScalar* rho, const aclScalar* epsilon,
    const aclTensor* grad, const aclTensor* varOut,
    const aclTensor* accumOut, const aclTensor* accumUpdateOut)
{
    OP_CHECK_NULL(var, return false);
    OP_CHECK_NULL(accum, return false);
    OP_CHECK_NULL(accumUpdate, return false);
    OP_CHECK_NULL(lr, return false);
    OP_CHECK_NULL(rho, return false);
    OP_CHECK_NULL(epsilon, return false);
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(varOut, return false);
    OP_CHECK_NULL(accumOut, return false);
    OP_CHECK_NULL(accumUpdateOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* var, const aclTensor* accum,
    const aclTensor* accumUpdate, const aclTensor* grad)
{
    auto dtype = var->GetDataType();
    if (!CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported dtype: %d. Only FLOAT and FLOAT16 are supported.",
                static_cast<int>(dtype));
        return false;
    }
    // All tensor dtypes must match
    OP_CHECK_DTYPE_NOT_MATCH(accum, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(accumUpdate, dtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(grad, dtype, return false);
    return true;
}

static bool CheckShapeConsistent(
    const aclTensor* var, const aclTensor* accum,
    const aclTensor* accumUpdate, const aclTensor* grad)
{
    OP_CHECK_MAX_DIM(var, ACLNN_MAX_SHAPE_RANK, return false);

    auto varShape = var->GetViewShape();
    auto accumShape = accum->GetViewShape();
    auto auShape = accumUpdate->GetViewShape();
    auto gradShape = grad->GetViewShape();

    // Reject 0-dim scalar tensors (REQUIREMENTS.md §5.3: supported shape dim 1~8).
    // See docs/aclnnApplyAdadelta.md "不支持 0 维标量 Tensor".
    if (varShape.GetDimNum() == 0 || accumShape.GetDimNum() == 0 ||
        auShape.GetDimNum() == 0 || gradShape.GetDimNum() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "0-dim scalar Tensor is not supported; supported dim range is 1~%d",
                ACLNN_MAX_SHAPE_RANK);
        return false;
    }

    if (varShape.GetDimNum() != accumShape.GetDimNum() ||
        varShape.GetDimNum() != auShape.GetDimNum() ||
        varShape.GetDimNum() != gradShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Shape dims mismatch: var=%zu, accum=%zu, accumUpdate=%zu, grad=%zu",
                varShape.GetDimNum(), accumShape.GetDimNum(),
                auShape.GetDimNum(), gradShape.GetDimNum());
        return false;
    }

    for (size_t i = 0; i < varShape.GetDimNum(); i++) {
        if (varShape.GetDim(i) != accumShape.GetDim(i) ||
            varShape.GetDim(i) != auShape.GetDim(i) ||
            varShape.GetDim(i) != gradShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Shape mismatch at dim %zu: var=%ld, accum=%ld, accumUpdate=%ld, grad=%ld",
                    i, varShape.GetDim(i), accumShape.GetDim(i),
                    auShape.GetDim(i), gradShape.GetDim(i));
            return false;
        }
    }
    return true;
}

// Validate scalar values:
//   - epsilon must be > 0 (numerical stability constant)
//   - rho must be in [0, 1) (decay coefficient, per REQUIREMENTS.md)
//   - lr must be >= 0 (learning rate; standard Adadelta requires non-negative lr.
//     Note: Under negative lr, NPU observations show var_out is left unchanged
//     while accum/accum_update are correctly updated, i.e. the computation is
//     ill-defined. The current hypothesis is related to the CANN ACLNN tiling
//     cache key missing scalar attrs; see docs/precision-report.md
//     "CANN ACLNN tiling 缓存键缺失 attr" section for background. We therefore
//     reject lr < 0 at the ACLNN layer as a conservative guard.)
static bool CheckScalarValues(const aclScalar* lr, const aclScalar* rho, const aclScalar* epsilon)
{
    float lrVal  = lr->ToFloat();
    float rhoVal = rho->ToFloat();
    float epsVal = epsilon->ToFloat();
    if (lrVal < 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "lr must be >= 0, got %f", lrVal);
        return false;
    }
    if (epsVal <= 0.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "epsilon must be > 0, got %f", epsVal);
        return false;
    }
    if (rhoVal < 0.0f || rhoVal >= 1.0f) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "rho must be in [0, 1), got %f", rhoVal);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* var, const aclTensor* accum, const aclTensor* accumUpdate,
    const aclScalar* lr, const aclScalar* rho, const aclScalar* epsilon,
    const aclTensor* grad, const aclTensor* varOut,
    const aclTensor* accumOut, const aclTensor* accumUpdateOut)
{
    if (!CheckNotNull(var, accum, accumUpdate, lr, rho, epsilon, grad, varOut, accumOut, accumUpdateOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Null pointer in input parameters");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(var, accum, accumUpdate, grad)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShapeConsistent(var, accum, accumUpdate, grad)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckScalarValues(lr, rho, epsilon)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyAdadeltaGetWorkspaceSize(
    const aclTensor* var,
    const aclTensor* accum,
    const aclTensor* accumUpdate,
    const aclScalar* lr,
    const aclScalar* rho,
    const aclScalar* epsilon,
    const aclTensor* grad,
    aclTensor* varOut,
    aclTensor* accumOut,
    aclTensor* accumUpdateOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnApplyAdadelta,
                    DFX_IN(var, accum, accumUpdate, grad),
                    DFX_OUT(varOut, accumOut, accumUpdateOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(var, accum, accumUpdate, lr, rho, epsilon, grad, varOut, accumOut, accumUpdateOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (var->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make inputs contiguous
    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto accumContiguous = l0op::Contiguous(accum, uniqueExecutor.get());
    CHECK_RET(accumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto accumUpdateContiguous = l0op::Contiguous(accumUpdate, uniqueExecutor.get());
    CHECK_RET(accumUpdateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Extract scalar values and pass to L0 API
    float lrVal = lr->ToFloat();
    float rhoVal = rho->ToFloat();
    float epsVal = epsilon->ToFloat();

    // Call L0 API
    auto opResults = l0op::ApplyAdadelta(
        varContiguous, accumContiguous, accumUpdateContiguous, gradContiguous,
        lrVal, rhoVal, epsVal,
        uniqueExecutor.get());
    CHECK_RET(opResults.varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy for all inplace outputs
    auto viewCopy1 = l0op::ViewCopy(opResults.varOut, varOut, uniqueExecutor.get());
    CHECK_RET(viewCopy1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopy2 = l0op::ViewCopy(opResults.accumOut, accumOut, uniqueExecutor.get());
    CHECK_RET(viewCopy2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopy3 = l0op::ViewCopy(opResults.accumUpdateOut, accumUpdateOut, uniqueExecutor.get());
    CHECK_RET(viewCopy3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApplyAdadelta(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyAdadelta);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
