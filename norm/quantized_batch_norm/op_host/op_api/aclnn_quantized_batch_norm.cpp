/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quantized_batch_norm.cpp
 * \brief
 */
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "../../../norm_common/op_host/norm_tensor_util.h"
#include "quantized_batch_norm.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/fill.h"
#include "aclnn_kernels/reshape.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/unsqueeze.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_quantized_batch_norm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t QBN2D_INPUT_DIMS = 4;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_CURRENT = {
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* input, const aclTensor* out, const aclTensor* mean, const aclTensor* var)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(var, return false);
    return true;
}

static bool CheckScalarNotNull(
    const aclScalar* inputScale, const aclScalar* inputZeroPoint, const aclScalar* outputScale,
    const aclScalar* outputZeroPoint)
{
    OP_CHECK_NULL(inputScale, return false);
    OP_CHECK_NULL(inputZeroPoint, return false);
    OP_CHECK_NULL(outputScale, return false);
    OP_CHECK_NULL(outputZeroPoint, return false);
    return true;
}

static aclnnStatus CheckNotEmpty(const aclTensor* input, const aclTensor* mean, const aclTensor* var)
{
    CHECK_RET(!input->IsEmpty(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(!mean->IsEmpty(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(!var->IsEmpty(), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOthersNotEmpty(const aclTensor* weight, const aclTensor* bias)
{
    if (weight != nullptr) {
        CHECK_RET(!weight->IsEmpty(), ACLNN_ERR_PARAM_INVALID);
    }

    if (bias != nullptr) {
        CHECK_RET(!bias->IsEmpty(), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST_CURRENT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_CURRENT, return false);

    return true;
}

static bool CheckOtherDtypeValid(
    const aclTensor* weight, const aclTensor* bias, const aclTensor* mean, const aclTensor* var)
{
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(weight, op::DataType::DT_FLOAT, return false);
    }
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(bias, op::DataType::DT_FLOAT, return false);
    }
    if (mean != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(mean, op::DataType::DT_FLOAT, return false);
    }
    if (var != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(var, op::DataType::DT_FLOAT, return false);
    }
    return true;
}

static bool CheckScalarDtypeValid(
    const aclScalar* inputScale, const aclScalar* inputZeroPoint, const aclScalar* outputScale,
    const aclScalar* outputZeroPoint)
{
    if (inputScale != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputScale, op::DataType::DT_FLOAT, return false);
    }
    if (inputZeroPoint != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputZeroPoint, op::DataType::DT_INT32, return false);
    }
    if (outputScale != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputScale, op::DataType::DT_FLOAT, return false);
    }
    if (outputZeroPoint != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputZeroPoint, op::DataType::DT_INT32, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor* input, const aclTensor* out)
{
    if (input->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, input [%s], output [%s].",
            op::ToString(input->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW.");
        return false;
    }
    return true;
}

static bool CheckOtherShape(
    int dimC, const aclTensor* weight, const aclTensor* bias, const aclTensor* mean, const aclTensor* var)
{
    if (weight != nullptr && (weight->GetViewShape().GetDimNum() != 1 || weight->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of weight should be one and shape is channel num of input.");
        return false;
    }
    if (bias != nullptr && (bias->GetViewShape().GetDimNum() != 1 || bias->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of bias should be one and shape is channel num of input.");
        return false;
    }
    if (mean != nullptr && (mean->GetViewShape().GetDimNum() != 1 || mean->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of mean should be one and shape is channel num of input.");
        return false;
    }
    if (var != nullptr && (var->GetViewShape().GetDimNum() != 1 || var->GetViewShape()[0] != dimC)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of var should be one and shape is channel num of input.");
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* input, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(out, input, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* output)
{
    CHECK_RET(CheckDtypeValid(input, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(input, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(input, output), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOthersParams(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, const aclTensor* mean, const aclTensor* var)
{
    CHECK_RET(CheckOtherDtypeValid(weight, bias, mean, var), ACLNN_ERR_PARAM_INVALID);
    int dimC = input->GetViewShape()[1];
    CHECK_RET(CheckOtherShape(dimC, weight, bias, mean, var), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckScalarParams(
    const aclScalar* inputScale, const aclScalar* inputZeroPoint, const aclScalar* outputScale,
    const aclScalar* outputZeroPoint)
{
    CHECK_RET(CheckScalarDtypeValid(inputScale, inputZeroPoint, outputScale, outputZeroPoint), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus QuantizedBatchNormProc2(
    const aclTensor* input, const aclTensor* mean, const aclTensor* var, const aclScalar* inputScale,
    const aclScalar* inputZeroPoint, const aclScalar* outputScale, const aclScalar* outputZeroPoint, aclTensor* weight,
    aclTensor* bias, float epsilon, aclTensor** output, aclOpExecutor* executor)
{
    op::DataType weightBiasDstType = DataType::DT_FLOAT;

    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightCast = l0op::Cast(weightContiguous, weightBiasDstType, executor);
    CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasContiguous = l0op::Contiguous(bias, executor);
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto biasCast = l0op::Cast(biasContiguous, weightBiasDstType, executor);
    CHECK_RET(biasCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanContiguous = l0op::Contiguous(mean, executor);
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanCast = l0op::Cast(meanContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto varContiguous = l0op::Contiguous(var, executor);
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto varCast = l0op::Cast(varContiguous, DataType::DT_FLOAT, executor);
    CHECK_RET(varCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputScaleTensor = executor->ConvertToTensor(inputScale, DataType::DT_FLOAT);
    CHECK_RET(inputScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputZeroPointTensor = executor->ConvertToTensor(inputZeroPoint, DataType::DT_INT32);
    CHECK_RET(inputZeroPointTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outputScaleTensor = executor->ConvertToTensor(outputScale, DataType::DT_FLOAT);
    CHECK_RET(outputScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outputZeroPointTensor = executor->ConvertToTensor(outputZeroPoint, DataType::DT_INT32);
    CHECK_RET(outputZeroPointTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    l0op::QuantizedBatchNormParams params{
        .x = input,
        .mean = meanCast,
        .var = varCast,
        .inputScale = inputScaleTensor,
        .inputZeroPoint = inputZeroPointTensor,
        .outputScale = outputScaleTensor,
        .outputZeroPoint = outputZeroPointTensor,
        .weight = weightCast,
        .bias = biasCast};
    *output = const_cast<aclTensor*>(l0op::QuantizedBatchNorm(params, epsilon, executor));

    return ACLNN_SUCCESS;
}

aclnnStatus QuantizedBatchNormProc1(
    const aclTensor* input, const aclTensor* mean, const aclTensor* var, const aclScalar* inputScale,
    const aclScalar* inputZeroPoint, const aclScalar* outputScale, const aclScalar* outputZeroPoint, aclTensor* weight,
    aclTensor* bias, float epsilon, aclTensor** output, aclOpExecutor* executor)
{
    size_t dimC = input->GetViewShape()[1];

    CHECK_RET(mean != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(var != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(input != nullptr, ACLNN_ERR_PARAM_INVALID);

    if (weight == nullptr) {
        weight = op::FillScalar(dimC, 1, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (bias == nullptr) {
        bias = op::FillScalar(dimC, 0, executor);
        CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    size_t dimNum = input->GetViewShape().GetDimNum();
    CHECK_RET(dimNum == QBN2D_INPUT_DIMS, ACLNN_ERR_PARAM_INVALID);

    auto inputPre = input;
    CHECK_RET(inputPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclTensor* result = nullptr;
    CHECK_RET(
        QuantizedBatchNormProc2(
            inputPre, mean, var, inputScale, inputZeroPoint, outputScale, outputZeroPoint, weight, bias, epsilon,
            &result, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);

    *output = result;
    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(
    const aclTensor* input, const aclTensor* mean, const aclTensor* var, const aclScalar* inputScale,
    const aclScalar* inputZeroPoint, const aclScalar* outputScale, const aclScalar* outputZeroPoint, aclTensor* weight,
    aclTensor* bias, float epsilon, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnQuantizedBatchNorm,
        DFX_IN(input, mean, var, inputScale, inputZeroPoint, outputScale, outputZeroPoint, weight, bias, epsilon),
        DFX_OUT(output));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(input, output, mean, var), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckScalarNotNull(inputScale, inputZeroPoint, outputScale, outputZeroPoint), ACLNN_ERR_PARAM_NULLPTR);

    auto retEmpty = CheckNotEmpty(input, mean, var);
    auto retOtherEmpty = CheckOthersNotEmpty(weight, bias);
    CHECK_RET(retEmpty == ACLNN_SUCCESS, retEmpty);
    CHECK_RET(retOtherEmpty == ACLNN_SUCCESS, retOtherEmpty);

    auto ret1 = CheckParams(input, output);
    CHECK_RET(ret1 == ACLNN_SUCCESS, ret1);
    auto ret2 = CheckOthersParams(input, weight, bias, mean, var);
    CHECK_RET(ret2 == ACLNN_SUCCESS, ret2);
    auto ret3 = CheckScalarParams(inputScale, inputZeroPoint, outputScale, outputZeroPoint);
    CHECK_RET(ret3 == ACLNN_SUCCESS, ret3);

    const aclTensor* inputContiguous = nullptr;
    if (input != nullptr) {
        inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto inputShape = input->GetViewShape();
    auto inputDims = inputShape.GetDimNum();
    CHECK_RET(inputDims == QBN2D_INPUT_DIMS, ACLNN_ERR_PARAM_INVALID);

    aclTensor* qbnOutput = nullptr;
    auto bnResult = QuantizedBatchNormProc1(
        inputContiguous, mean, var, inputScale, inputZeroPoint, outputScale, outputZeroPoint, weight, bias, epsilon,
        &qbnOutput, uniqueExecutor.get());
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    if (qbnOutput != nullptr) {
        auto viewCopyResult = l0op::ViewCopy(qbnOutput, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantizedBatchNorm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantizedBatchNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif