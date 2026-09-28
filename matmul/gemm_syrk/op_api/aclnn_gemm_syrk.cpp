/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gemm_syrk.h"

#include <cmath>
#include <cstring>

#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/muls.h"
#include "gemm_syrk.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/log_format_util.h"

using namespace op;
using namespace Ops::NN;
using Ops::NN::FormatString;

namespace {
constexpr size_t MIN_DIM_NUM = 2;
constexpr size_t MAX_DIM_NUM = 6;
constexpr float GEMM_SYRK_DEFAULT_SCALE_VALUE = 1.0F;

struct GemmSyrkParams {
    const aclTensor* a{nullptr};
    aclTensor* cRef{nullptr};
    const aclScalar* alphaOptional{nullptr};
    const aclScalar* betaOptional{nullptr};
    bool transposeX{false};
    const char* fillMode{nullptr};
};

static aclnnStatus CheckNotNull(const GemmSyrkParams& params)
{
    OP_CHECK_NULL(params.a, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.cRef, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static bool CheckScaleParam(const aclScalar* scaleOptional, const char* name)
{
    if (scaleOptional == nullptr) {
        return true;
    }
    const auto scaleType = scaleOptional->GetDataType();
    if (scaleType != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s only supports float32.", name);
        return false;
    }
    return true;
}

static aclnnStatus CheckDtype(const GemmSyrkParams& params)
{
    auto aType = params.a->GetDataType();
    if (aType != DataType::DT_FLOAT16 && aType != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a only supports float16 or bfloat16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cRef->GetDataType() != aType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a and c must have the same dtype.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckScaleParam(params.alphaOptional, "alphaOptional") ||
        !CheckScaleParam(params.betaOptional, "betaOptional")) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const GemmSyrkParams& params)
{
    if (params.a->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cRef->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "c only supports ND format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const GemmSyrkParams& params)
{
    const auto& aShape = params.a->GetViewShape();
    const auto& cShape = params.cRef->GetViewShape();
    const int64_t aDimNum = aShape.GetDimNum();
    const int64_t cDimNum = cShape.GetDimNum();
    if (aDimNum < static_cast<int64_t>(MIN_DIM_NUM) || aDimNum > static_cast<int64_t>(MAX_DIM_NUM) ||
        aDimNum != cDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "the dims of a and c must be the same and within the range [2, 6], got a %ldD, c %ldD.", aDimNum,
                cDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // transposeX = false: a is [..., m, k]; transposeX = true: a is the
    // transposed [..., k, m] storage. c is the square [..., m, m] in both
    // cases; the batch-axis must match (no in-place broadcast).
    for (int64_t i = 0; i + MIN_DIM_NUM < aDimNum; ++i) {
        if (aShape.GetDim(i) != cShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the batch-axis of a and c must be the same.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t m = params.transposeX ? aShape.GetDim(aDimNum - 1) : aShape.GetDim(aDimNum - 2);
    if (cShape.GetDim(cDimNum - MIN_DIM_NUM) != m || cShape.GetDim(cDimNum - 1) != m) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "a must be [..., m, k] (or [..., k, m] with transposeX) and c must be the square matrix "
                "[..., m, m], got m %ld.",
                m);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (m < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the m-axis of a must not be a negative number.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFillMode(const GemmSyrkParams& params)
{
    const char* fillMode = params.fillMode == nullptr ? "full" : params.fillMode;
    if (strcmp(fillMode, "full") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "fillMode only supports \"full\" currently, \"up\"/\"low\" are not implemented, got %s.", fillMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const GemmSyrkParams& params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFillMode(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (!IsNpuArch3510Series()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GemmSyrk only supports Ascend 950 / 350 (DAV_3510).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GemmSyrkGetWorkspaceSizeCommon(const GemmSyrkParams& params, aclOpExecutor* executor)
{
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const float alphaValue = params.alphaOptional == nullptr ? GEMM_SYRK_DEFAULT_SCALE_VALUE :
                                                               params.alphaOptional->ToFloat();
    const float betaValue = params.betaOptional == nullptr ? GEMM_SYRK_DEFAULT_SCALE_VALUE :
                                                             params.betaOptional->ToFloat();

    const auto& aShape = params.a->GetViewShape();
    const int64_t aDimNum = aShape.GetDimNum();
    const int64_t m = params.transposeX ? aShape.GetDim(aDimNum - 1) : aShape.GetDim(aDimNum - MIN_DIM_NUM);
    const int64_t k = params.transposeX ? aShape.GetDim(aDimNum - MIN_DIM_NUM) : aShape.GetDim(aDimNum - 1);
    int64_t batch = 1;
    for (int64_t i = 0; i + MIN_DIM_NUM < aDimNum; ++i) {
        batch *= aShape.GetDim(i);
    }
    if (m == 0 || batch == 0) {
        OP_LOGD("empty tensor, m or batch is 0, nothing to do");
        return ACLNN_SUCCESS;
    }

    // The Blaze kernel reads contiguous ND buffers; Contiguous is a no-op for
    // already-contiguous inputs.
    auto aContig = l0op::Contiguous(params.a, executor);
    CHECK_RET(aContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cContig = l0op::Contiguous(params.cRef, executor);
    CHECK_RET(cContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (k == 0 || alphaValue == 0.0F) {
        // A @ A^T is the m x m zero matrix: C = beta * C in place.
        auto scaled = l0op::Muls(cContig, betaValue, executor);
        CHECK_RET(scaled != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(scaled, params.cRef, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    // The in-place addend tensor is bound as both the op input and output; drop
    // the const the same way l0op::GemmV3Nd aliases the user's c tensor.
    auto result = l0op::GemmSyrk(aContig, const_cast<aclTensor*>(cContig), alphaValue, betaValue, params.transposeX,
                                 params.fillMode == nullptr ? "full" : params.fillMode, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // result aliases cContig; bind the graph output back to the user tensor
    // (a self-copy no-op when cRef is already the contiguous buffer).
    auto viewCopyResult = l0op::ViewCopy(result, params.cRef, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGemmSyrkGetWorkspaceSize(const aclTensor* a, aclTensor* cRef, const aclScalar* alphaOptional,
                                          const aclScalar* betaOptional, bool transposeX, const char* fillMode,
                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    GemmSyrkParams params{a, cRef, alphaOptional, betaOptional, transposeX, fillMode};

    L2_DFX_PHASE_1(aclnnGemmSyrk, DFX_IN(a, cRef, alphaOptional, betaOptional), DFX_OUT(cRef));

    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = GemmSyrkGetWorkspaceSizeCommon(params, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGemmSyrk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGemmSyrk);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GemmSyrk launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
