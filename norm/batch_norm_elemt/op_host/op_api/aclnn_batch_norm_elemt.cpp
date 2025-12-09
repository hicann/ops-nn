/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "../../../norm_common/op_host/norm_tensor_util.h"
#include "level0/mul.h"
#include "level0/div.h"
#include "level0/sub.h"
#include "level0/ones_like.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_batch_norm_elemt.h"

using namespace op;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* input, aclTensor* mean, aclTensor* invstd, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(invstd, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

template <typename T, typename... Ts>
static bool CheckDtypeValid(const T& t, const Ts&... args)
{
    if constexpr (std::is_same_v<T, aclTensor*> || std::is_same_v<T, const aclTensor*>) {
        bool isAscend910BSocVersion =
            (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
             GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
        const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST =
            isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;
        if (t && !CheckType(t->GetDataType(), DTYPE_SUPPORT_LIST)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "input not implemented for %s, should be in dtype support list %s.",
                op::ToString(t->GetDataType()).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
            return false;
        }
    }

    if constexpr (sizeof...(args) > 0) {
        return CheckDtypeValid(args...);
    }
    return true;
}

static bool CheckFormat(const aclTensor* input, const aclTensor* out)
{
    if (input->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and out should be equal, input [%s], out [%s].",
            op::ToString(input->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND,NCHW,NCL,NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, const aclTensor* mean,
    const aclTensor* invstd, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(input, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(out, BN_MIN_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(out, input, return false);
    if (input->GetViewShape()[1] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input channel size should not be zero.");
        return false;
    }

    if ((weight != nullptr && weight->GetViewShape().GetDimNum() != 1) ||
        (bias != nullptr && bias->GetViewShape().GetDimNum() != 1) || mean->GetViewShape().GetDimNum() != 1 ||
        invstd->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight/bias/mean/invstd dim size should be one.");
        return false;
    }

    int64_t dimC = input->GetViewShape()[1];
    if ((weight != nullptr && weight->GetViewShape()[0] != dimC) ||
        (bias != nullptr && bias->GetViewShape()[0] != dimC) || mean->GetViewShape()[0] != dimC ||
        invstd->GetViewShape()[0] != dimC) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight/bias/mean/invstd shape should be same as the channel size of input.");
        return false;
    }
    return true;
}

static aclnnStatus CheckBatchNormElemtParams(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* mean, aclTensor* invstd,
    const aclTensor* output)
{
    CHECK_RET(CheckNotNull(input, mean, invstd, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(input, weight, bias, mean, invstd, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(input, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(input, weight, bias, mean, invstd, output), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormElemtGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* mean, aclTensor* invstd,
    double eps, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBatchNormElemt, DFX_IN(input, weight, bias, mean, invstd, eps), DFX_OUT(output));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckBatchNormElemtParams(input, weight, bias, mean, invstd, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input != nullptr && input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto inputShape = input->GetViewShape();
    auto inputDims = inputShape.GetDimNum();

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const size_t maxDims = 5;
    if (inputDims > maxDims) {
        const int64_t shapes[5] = {inputShape[0], inputShape[1], inputShape[2], inputShape[3], -1};
        aclIntArray* shapeArray = uniqueExecutor.get()->AllocIntArray(shapes, 5);
        inputContiguous = l0op::Reshape(inputContiguous, shapeArray, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputContiguous = l0op::ReFormat(inputContiguous, Format::FORMAT_NCDHW);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    size_t dimC = inputShape[1];
    if (weight == nullptr) {
        weight = FillScalar(dimC, 1, uniqueExecutor.get());
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (bias == nullptr) {
        bias = FillScalar(dimC, 0, uniqueExecutor.get());
        CHECK_RET(bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto invstdContiguous = l0op::Contiguous(invstd, uniqueExecutor.get());
    CHECK_RET(invstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto variance = l0op::Mul(invstdContiguous, invstdContiguous, uniqueExecutor.get());
    CHECK_RET(variance != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto one = l0op::OnesLike(variance, uniqueExecutor.get());
    CHECK_RET(one != nullptr, ACLNN_ERR_INNER_NULLPTR);

    variance = l0op::Div(one, variance, uniqueExecutor.get());
    CHECK_RET(variance != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto epsScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(eps));
    auto epsTensor = uniqueExecutor.get()->ConvertToTensor(epsScalar, variance->GetDataType());

    variance = l0op::Sub(variance, epsTensor, uniqueExecutor.get());
    CHECK_RET(variance != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanNonConst = const_cast<aclTensor*>(meanContiguous);
    auto varianceNonConst = const_cast<aclTensor*>(variance);
    aclTensor* bnOutput = nullptr;
    auto bnResult = BatchNorm(
        inputContiguous, weightContiguous, biasContiguous, meanNonConst, varianceNonConst, false, 0.0, eps, &bnOutput,
        nullptr, nullptr, uniqueExecutor.get());
    CHECK_RET(bnResult == ACLNN_SUCCESS, bnResult);

    if (inputDims > maxDims) {
        int64_t originShapes[inputDims];
        for (size_t i = 0; i < inputDims; ++i) {
            originShapes[i] = inputShape[i];
        }
        aclIntArray* originShapeArray = uniqueExecutor.get()->AllocIntArray(originShapes, inputDims);
        auto bnOutputReshape = l0op::Reshape(bnOutput, originShapeArray, uniqueExecutor.get());
        auto bnOutputReformat = l0op::ReFormat(bnOutputReshape, Format::FORMAT_ND);
        auto viewCopyResult = l0op::ViewCopy(bnOutputReformat, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto viewCopyResult = l0op::ViewCopy(bnOutput, output, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormElemt(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormElemt);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
