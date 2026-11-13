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
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "level0/muls.h"
#include "level0/zero_op.h"
#include "gemm_syrk.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
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
    int8_t cubeMathType{0};
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
    if (aType != DataType::DT_FLOAT16 && aType != DataType::DT_BF16 && aType != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a only supports float16, bfloat16 or float32 (HF32 cube mode).");
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

static bool IsValidLinearView(const aclTensor* tensor)
{
    if (tensor->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        return false;
    }
    const auto& viewShape = tensor->GetViewShape();
    const auto& viewStrides = tensor->GetViewStrides();
    const int64_t dimNum = viewShape.GetDimNum();
    if (dimNum <= 0 || static_cast<int64_t>(viewStrides.size()) != dimNum) {
        return false;
    }
    int64_t maxOffset = tensor->GetViewOffset();
    for (int64_t i = 0; i < dimNum; ++i) {
        const int64_t dim = viewShape.GetDim(i);
        if (dim < 0 || viewStrides[i] < 0) {
            return false; // 负维度/负 stride 的描述符非法
        }
        if (dim > 0) {
            maxOffset += (dim - 1) * viewStrides[i];
        }
    }
    if (viewShape.GetShapeSize() == 0) {
        return true; // 空张量不访问数据, 调用侧已按 no-op 处理
    }
    const auto& storageShape = tensor->GetStorageShape();
    if (storageShape.GetDimNum() == 0) {
        // storageDims 缺省(aclCreateTensor 传 nullptr 时 storageShape 为空): 描述符
        // 不携带真实 buffer 容量, 无法判越界, 只保留上面的自洽性判定
        return true;
    }
    // storageDims 已显式声明(含与 viewShape 同形): 声明的容量是有效信息, 一律校验。
    // 同形 + 连续/offset=0 的稠密张量 maxOffset = shapeSize - 1 < shapeSize 恒过;
    // 同形但 viewOffset 非零或 stride 跨度超出声明容量的描述符在此被拒绝
    return storageShape.GetShapeSize() > maxOffset;
}

static aclnnStatus CheckFormat(const GemmSyrkParams& params)
{
    if (!IsValidLinearView(params.a)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "a must be a valid view of a linear (ND) storage: shape %s, strides %s, storage format %s.",
                op::ToString(params.a->GetViewShape()).GetString(),
                op::ToString(params.a->GetViewStrides()).GetString(),
                op::ToString(params.a->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsValidLinearView(params.cRef)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "c must be a valid view of a linear (ND) storage: shape %s, strides %s, storage format %s.",
                op::ToString(params.cRef->GetViewShape()).GetString(),
                op::ToString(params.cRef->GetViewStrides()).GetString(),
                op::ToString(params.cRef->GetStorageFormat()).GetString());
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

static aclnnStatus CheckCubeMathType(const GemmSyrkParams& params)
{
    if (params.cubeMathType < KEEP_DTYPE || params.cubeMathType > USE_HF32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "cubeMathType only supports 0 (KEEP_DTYPE), 1 (ALLOW_FP32_DOWN_PRECISION), 2 (USE_FP16), "
                "3 (USE_HF32), but got %d.",
                static_cast<int32_t>(params.cubeMathType));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.a->GetDataType() == DataType::DT_FLOAT && params.cubeMathType == KEEP_DTYPE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "For float32 input, cubeMathType 0 (KEEP_DTYPE) is not supported; use 1/3 for HF32 or 2 for FP16 compute.");
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
    CHECK_RET(CheckCubeMathType(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (!IsNpuArch3510Series()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GemmSyrk only supports Ascend 950 / 350 (DAV_3510).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GemmSyrkGetWorkspaceSizeCommon(const GemmSyrkParams& params, aclOpExecutor* executor)
{
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

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

    aContig = l0op::ReFormat(aContig, op::Format::FORMAT_ND, executor);
    CHECK_RET(aContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    cContig = l0op::ReFormat(cContig, op::Format::FORMAT_ND, executor);
    CHECK_RET(cContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (k == 0 || alphaValue == 0.0F) {
        // A @ A^T is the m x m zero matrix: C = beta * C in place.
        if (betaValue != 0.0F) {
            auto scaled = l0op::Muls(cContig, betaValue, executor);
            CHECK_RET(scaled != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyResult = l0op::ViewCopy(scaled, params.cRef, executor);
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            // cuBLAS contract: beta == 0 means C is not read at all ("C does
            // not have to be a valid input"), and the A @ A^T term is zero, so
            // the output is all zeros regardless of (possibly NaN/Inf) C.
            auto zeros = l0op::ZerosLike(cContig, executor);
            CHECK_RET(zeros != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyResult = l0op::ViewCopy(zeros, params.cRef, executor);
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        return ACLNN_SUCCESS;
    }

    // FP32 + USE_FP16: cast the operands to FP16, run the regular FP16 kernel
    // path, then cast the result back to FP32. The user-visible cRef stays
    // FLOAT32; only the cube compute precision changes.
    const bool useFp16Compute = params.a->GetDataType() == DataType::DT_FLOAT && params.cubeMathType == USE_FP16;
    const aclTensor* opA = aContig;
    const aclTensor* opC = cContig;
    if (useFp16Compute) {
        opA = l0op::Cast(aContig, DataType::DT_FLOAT16, executor);
        CHECK_RET(opA != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opC = l0op::Cast(cContig, DataType::DT_FLOAT16, executor);
        CHECK_RET(opC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // The in-place addend tensor is bound as both the op input and output; drop
    // the const the same way l0op::GemmV3Nd aliases the user's c tensor.
    auto result = l0op::GemmSyrk(opA, const_cast<aclTensor*>(opC), alphaValue, betaValue, params.transposeX,
                                 params.fillMode == nullptr ? "full" : params.fillMode, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* opResult = result;
    if (useFp16Compute) {
        opResult = l0op::Cast(result, DataType::DT_FLOAT, executor);
        CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // result aliases cContig; bind the graph output back to the user tensor
    // (a self-copy no-op when cRef is already the contiguous buffer).
    auto viewCopyResult = l0op::ViewCopy(opResult, params.cRef, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGemmSyrkGetWorkspaceSize(const aclTensor* a, aclTensor* cRef, const aclScalar* alphaOptional,
                                          const aclScalar* betaOptional, bool transposeX, const char* fillMode,
                                          int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    GemmSyrkParams params{a, cRef, alphaOptional, betaOptional, transposeX, fillMode, cubeMathType};

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
