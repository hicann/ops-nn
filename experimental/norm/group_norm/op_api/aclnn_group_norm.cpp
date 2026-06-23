/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_group_norm_experimental.h"
#include "group_norm_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/ones_like.h"
#include "level0/zero_op.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <tuple>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* out,
    const aclTensor* meanOut, const aclTensor* rstdOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    if (gamma != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(gamma, self->GetDataType(), return false);
    }
    if (beta != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(beta, self->GetDataType(), return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(meanOut, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(rstdOut, self->GetDataType(), return false);
    return true;
}

static int64_t GetTensorNumel(const aclTensor* x, size_t startIdx)
{
    int64_t numel = 1;
    for (size_t i = startIdx; i < x->GetViewShape().GetDimNum(); ++i) {
        numel *= x->GetViewShape().GetDim(i);
    }
    return numel;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t n, int64_t c, int64_t hxw,
    int64_t group, const aclTensor* out, const aclTensor* meanOut, const aclTensor* rstdOut)
{
    OP_CHECK_MIN_DIM(self, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, op::Shape({n, group}), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(rstdOut, op::Shape({n, group}), return false);
    OP_CHECK(
        self->GetViewShape()[0] == n, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self.shape[0] must equal N."), return false);
    OP_CHECK(
        self->GetViewShape()[1] == c, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self.shape[1] must equal C."), return false);
    OP_CHECK(
        GetTensorNumel(self, 2) == hxw, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tail numel must equal HxW."), return false);
    if (gamma != nullptr) {
        OP_CHECK(
            gamma->GetViewShape().GetDimNum() == 1 && gamma->GetViewShape()[0] == c,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma shape should be [C]."), return false);
    }
    if (beta != nullptr) {
        OP_CHECK(
            beta->GetViewShape().GetDimNum() == 1 && beta->GetViewShape()[0] == c,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "beta shape should be [C]."), return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t n, int64_t c, int64_t hxw,
    int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut)
{
    CHECK_RET(CheckNotNull4Tensor(self, out, meanOut, rstdOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, gamma, beta, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(group > 0 && c % group == 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(eps > 0.0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, gamma, beta, n, c, hxw, group, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetGamma(const aclTensor* self, const aclTensor* gamma, int64_t c, aclOpExecutor* executor)
{
    if (gamma != nullptr) {
        return l0op::Contiguous(gamma, executor);
    }
    auto gammaTensor = executor->AllocTensor(op::Shape({c}), self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(gammaTensor != nullptr, nullptr);
    return l0op::OnesLike(gammaTensor, executor);
}

static const aclTensor* GetBeta(const aclTensor* self, const aclTensor* beta, int64_t c, aclOpExecutor* executor)
{
    if (beta != nullptr) {
        return l0op::Contiguous(beta, executor);
    }
    auto betaTensor = executor->AllocTensor(op::Shape({c}), self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(betaTensor != nullptr, nullptr);
    return l0op::ZerosLike(betaTensor, executor);
}

aclnnStatus aclnnGroupNormGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gammaOptional, const aclTensor* betaOptional, int64_t n, int64_t c,
    int64_t hxw, int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnGroupNorm, DFX_IN(self, gammaOptional, betaOptional, n, c, hxw, group, eps),
        DFX_OUT(out, meanOut, rstdOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(self, gammaOptional, betaOptional, n, c, hxw, group, eps, out, meanOut, rstdOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gammaContiguous = GetGamma(selfContiguous, gammaOptional, c, uniqueExecutor.get());
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto betaContiguous = GetBeta(selfContiguous, betaOptional, c, uniqueExecutor.get());
    CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::GroupNorm(
        selfContiguous, gammaContiguous, betaContiguous, n, group, static_cast<float>(eps), uniqueExecutor.get());
    auto y = std::get<0>(result);
    auto mean = std::get<1>(result);
    auto rstd = std::get<2>(result);
    CHECK_RET(y != nullptr && mean != nullptr && rstd != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(l0op::ViewCopy(y, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(mean, meanOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(rstd, rstdOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupNorm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
