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
 * @file aclnn_fast_gelu_v2.cpp
 * @brief ACLNN L2 API implementation for FastGeluV2
 */

#include "aclnn_fast_gelu_v2.h"
#include "fast_gelu_v2.h"
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
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool HasEmptyTensor(const aclTensor* x)
{
    return x->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);

    if (!CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Dtype not supported: dtype=%d. Supported: FLOAT, FLOAT16, BF16.",
                static_cast<int>(x->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* out)
{
    auto formatX = x->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    if (IsPrivateFormat(formatX) || IsPrivateFormat(formatOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Private format not supported: x=%d, out=%d",
                static_cast<int>(formatX), static_cast<int>(formatOut));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    // Verify output shape matches input shape (elementwise: y.shape == x.shape)
    auto xShape = x->GetViewShape();
    auto outShape = out->GetViewShape();
    if (xShape.GetDimNum() != outShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Shape rank mismatch: x has %zu dims, out has %zu dims",
                xShape.GetDimNum(), outShape.GetDimNum());
        return false;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        if (xShape.GetDim(i) != outShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Shape mismatch at dim %zu: x=%ld, out=%ld",
                    i, xShape.GetDim(i), outShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    if (!CheckNotNull(x, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed: x_dtype=%d, out_dtype=%d",
                static_cast<int>(x->GetDataType()), static_cast<int>(out->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckShape failed: x_dim=%zu, out_dim=%zu",
                x->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFastGeluV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFastGeluV2, DFX_IN(x), DFX_OUT(out));

    // Validate output parameters before any other work
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::FastGeluV2(xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFastGeluV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFastGeluV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
