/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_matmul_activation_quant_common.h"

#include <string>

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_matmul_activation_quant_checker.h"
#include "quant_matmul_activation_quant_util.h"

using namespace op;
using namespace QBMMActivationQuant;
using Ops::NN::FormatString;

namespace {

static aclnnStatus CheckInputRanks(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    if (x1DimNum < MIN_MATRIX_RANK || x1DimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "x1", FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the rank of x1 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MIN_MATRIX_RANK || x2DimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "x2", FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the rank of x2 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightNdInputs(const aclTensor* x1, const aclTensor* x2, const char* apiName)
{
    const auto status = CheckPlatformAndInputs(x1, x2, apiName);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    OP_LOGD("WeightND input validation succeeded.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    const auto shapeInfo = GetMatmulShapeInfo(params);
    CHECK_RET(CheckKDimensionsMatch(params, shapeInfo, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim, apiName), ACLNN_ERR_PARAM_INVALID);
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        CHECK_RET(CheckFp4Shape(params, shapeInfo, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckMxScaleLastDim(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckBiasShape(params, shapeInfo.nDim, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputScaleShapes(params, shapeInfo, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputShape(params, shapeInfo, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    OP_LOGD("Parameter validation started.");
    CHECK_RET(CheckDtype(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputFormats(params, Format::FORMAT_ND, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttributes(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Parameter validation succeeded.");

    return ACLNN_SUCCESS;
}

static aclnnStatus BuildExecutor(QuantMatmulActivationQuantParams& params, aclOpExecutor* executor, const char* apiName)
{
    CHECK_RET(CheckX3(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const bool transposeX1Before = params.transposeX1;
    CHECK_RET(PrepareCommonInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    auto reformattedX2 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x2);
    CHECK_RET(reformattedX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x2 = reformattedX2;
    const bool transposeX2Before = params.transposeX2;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.x2, params.transposeX2, executor),
              ACLNN_ERR_INNER_NULLPTR);
    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShapes(params, GetMatmulShapeInfo(params), x1ScaleExpectShape, x2ScaleExpectShape);
    auto ret = MxScaleContiguousProcess(params.x1Scale, params.transposeX1 != transposeX1Before, x1ScaleExpectShape,
                                        executor, "x1, x1Scale", apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MxScaleContiguousProcess(params.x2Scale, params.transposeX2 != transposeX2Before, x2ScaleExpectShape,
                                   executor, "x2, x2Scale", apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(CheckGroupSize(params, apiName), ACLNN_ERR_PARAM_INVALID);

    ret = CheckParams(params, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return AddQuantMatmulActivationQuantTasks(params, executor);
}

} // namespace

namespace QBMMActivationQuant {

aclnnStatus GetWorkspaceSize(const char* apiName, QuantMatmulActivationQuantParams params, uint64_t* workspaceSize,
                             aclOpExecutor** executor)
{
    auto ret = CheckWeightNdInputs(params.x1, params.x2, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    params.yDtype = params.x1->GetDataType();

    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckTensorViews(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputRanks(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(PrepareInputViews(params, executorPtr), ACLNN_ERR_INNER_NULLPTR);

    ret = BuildExecutor(params, executorPtr, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

} // namespace QBMMActivationQuant
