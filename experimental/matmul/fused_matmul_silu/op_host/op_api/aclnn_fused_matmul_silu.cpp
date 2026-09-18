/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_fused_matmul_silu.h"

#include "fused_matmul_silu.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

constexpr size_t kMatrixRank = 2;
constexpr size_t kBiasRank = 1;
constexpr int64_t kBlockK = 64;
constexpr int64_t kMaxSupportedK = 4096;

bool CheckNotNull(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

bool CheckDtype(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y)
{
    OP_CHECK(x->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be BF16."), return false);
    OP_CHECK(weight->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight must be BF16."),
             return false);
    OP_CHECK(bias->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias must be BF16."),
             return false);
    OP_CHECK(y->GetDataType() == DataType::DT_BF16, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y must be BF16."), return false);
    return true;
}

bool CheckFormat(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y)
{
    OP_CHECK(x->GetStorageFormat() == Format::FORMAT_ND, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be ND."),
             return false);
    OP_CHECK(weight->GetStorageFormat() == Format::FORMAT_ND, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight must be ND."),
             return false);
    OP_CHECK(bias->GetStorageFormat() == Format::FORMAT_ND, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias must be ND."),
             return false);
    OP_CHECK(y->GetStorageFormat() == Format::FORMAT_ND, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y must be ND."),
             return false);
    return true;
}

bool CheckShape(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y)
{
    OP_CHECK_MIN_DIM(x, kMatrixRank, return false);
    OP_CHECK_MAX_DIM(x, kMatrixRank, return false);
    OP_CHECK_MIN_DIM(weight, kMatrixRank, return false);
    OP_CHECK_MAX_DIM(weight, kMatrixRank, return false);
    OP_CHECK_MIN_DIM(y, kMatrixRank, return false);
    OP_CHECK_MAX_DIM(y, kMatrixRank, return false);
    OP_CHECK_MIN_DIM(bias, kBiasRank, return false);
    OP_CHECK_MAX_DIM(bias, kBiasRank, return false);

    const auto& xShape = x->GetViewShape();
    const auto& weightShape = weight->GetViewShape();
    const auto& biasShape = bias->GetViewShape();
    const auto& yShape = y->GetViewShape();
    OP_CHECK(xShape[0] > 0 && xShape[1] > 0 && weightShape[0] > 0 && weightShape[1] > 0 && biasShape[0] > 0 &&
                 yShape[0] > 0 && yShape[1] > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape dimensions must be positive."), return false);
    OP_CHECK(xShape[1] == weightShape[1], OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x K must equal weight K."), return false);
    OP_CHECK(biasShape[0] == weightShape[0], OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias shape must equal weight N."),
             return false);
    OP_CHECK(yShape[0] == xShape[0] && yShape[1] == weightShape[0],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape must be [M, N]."), return false);
    OP_CHECK(xShape[1] % kBlockK == 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "K must be aligned to 64."), return false);
    OP_CHECK(xShape[1] <= kMaxSupportedK, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "K must be less than or equal to 4096."),
             return false);
    return true;
}

aclnnStatus Validate(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(x, weight, bias, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(x, weight, bias, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, weight, bias, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, weight, bias, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnFusedMatmulSiluGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                            const aclTensor* bias, aclTensor* y,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMatmulSilu, DFX_IN(x, weight, bias), DFX_OUT(y));
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto validateRet = Validate(x, weight, bias, y);
    CHECK_RET(validateRet == ACLNN_SUCCESS, validateRet);

    const aclTensor* xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opOut = l0op::FusedMatmulSilu(xContiguous, weightContiguous, biasContiguous, uniqueExecutor.get());
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* copyOut = l0op::ViewCopy(opOut, y, uniqueExecutor.get());
    CHECK_RET(copyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnFusedMatmulSilu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMatmulSilu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
