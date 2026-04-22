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
 * @file aclnn_fused_bias_leaky_relu.cpp
 * @brief ACLNN L2 API implementation for FusedBiasLeakyRelu
 *
 * Two-phase design:
 * 1. aclnnFusedBiasLeakyReluGetWorkspaceSize - parameter checks, Contiguous, L0 call
 * 2. aclnnFusedBiasLeakyRelu - execute computation
 */

#include "aclnn_fused_bias_leaky_relu.h"
#include "fused_bias_leaky_relu.h"
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
    DataType::DT_FLOAT16, DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* x, const aclTensor* bias)
{
    return x->IsEmpty() || bias->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, bias->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);

    if (!IsDtypeSupported(x->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Dtype not supported: dtype=%d. Supported: FLOAT16, FLOAT.",
                static_cast<int>(x->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* bias, const aclTensor* out)
{
    auto formatX = x->GetStorageFormat();
    auto formatBias = bias->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    if (IsPrivateFormat(formatX) || IsPrivateFormat(formatBias) || IsPrivateFormat(formatOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Private format not supported: x=%d, bias=%d, out=%d",
                static_cast<int>(formatX), static_cast<int>(formatBias), static_cast<int>(formatOut));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(bias, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    // x and bias must have the same shape (no broadcast)
    auto xShape = x->GetViewShape();
    auto biasShape = bias->GetViewShape();
    if (xShape.GetDimNum() != biasShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x and bias dim mismatch: x_dim=%zu, bias_dim=%zu",
                xShape.GetDimNum(), biasShape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (xShape.GetDim(i) != biasShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "x and bias shape mismatch at dim %zu: x=%ld, bias=%ld",
                    i, xShape.GetDim(i), biasShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* bias, const aclTensor* out)
{
    if (!CheckNotNull(x, bias, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x, bias, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x, bias, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x, bias, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFusedBiasLeakyReluGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* bias,
    double negativeSlope,
    double scale,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedBiasLeakyRelu, DFX_IN(x, bias), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, bias, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x, bias)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::FusedBiasLeakyRelu(
        xContiguous, biasContiguous, negativeSlope, scale, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFusedBiasLeakyRelu(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedBiasLeakyRelu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
