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
    auto x2DimNum = params.x2->GetStorageShape().GetDimNum();
    const auto x2ViewDimNum = params.x2->GetViewShape().GetDimNum();
    if (x2ViewDimNum < MIN_MATRIX_RANK || x2ViewDimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "x2 view", FormatString("%zuD", x2ViewDimNum).c_str(),
                                                 "the logical rank of x2 must be in the range [2, 6]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1DimNum < MIN_MATRIX_RANK || x1DimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "x1", FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the rank of x1 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MIN_NZ_STORAGE_RANK || x2DimNum > MAX_NZ_STORAGE_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "x2", FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the rank of x2 must be in the range [4, 8]"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightNzInputs(const aclTensor* x1, const aclTensor* x2, const char* apiName)
{
    const auto status = CheckPlatformAndInputs(x1, x2, apiName);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(apiName, "x2", op::ToString(x2->GetStorageFormat()).GetString(),
                                               "the format of x2 must be FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (x2->GetDataType() == op::DataType::DT_FLOAT8_E5M2) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "x2", op::ToString(x2->GetDataType()).GetString(),
                                              "the FLOAT8_E5M2 dtype of x2 is supported only in ND format");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("WeightNZ input validation succeeded.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    const auto shapeInfo = GetMatmulShapeInfo(params);
    CHECK_RET(CheckKDimensionsMatch(params, shapeInfo, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // NZ情况下，x2的k和n不能为1
    int64_t dim1 = params.x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = params.x2->GetViewShape().GetDimNum() - LAST_SECOND_DIM_INDEX;
    if (params.x2->GetViewShape().GetDim(dim2) == 1 || params.x2->GetViewShape().GetDim(dim1) == 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, params.transposeX2 ? "x2 N, x2 K" : "x2 K, x2 N",
            FormatString("%lld, %lld", static_cast<long long>(params.x2->GetViewShape().GetDim(dim2)),
                         static_cast<long long>(params.x2->GetViewShape().GetDim(dim1)))
                .c_str(),
            "when the format of x2 is FRACTAL_NZ, the K and N dimensions of x2 cannot be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim, apiName), ACLNN_ERR_PARAM_INVALID);
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        if (params.transposeX1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "transposeX1", "true",
                "when the format of x2 is FRACTAL_NZ and the dtypes of x1, x2 and y are FP4, x1 cannot be transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
    CHECK_RET(CheckInputFormats(params, Format::FORMAT_FRACTAL_NZ, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttributes(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Parameter validation succeeded.");

    return ACLNN_SUCCESS;
}

static aclnnStatus BuildExecutor(QuantMatmulActivationQuantParams& params, bool transposeX2View,
                                 aclOpExecutor* executor, const char* apiName)
{
    CHECK_RET(CheckX3(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const bool transposeX1Before = params.transposeX1;
    CHECK_RET(PrepareCommonInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShapes(params, GetMatmulShapeInfo(params), x1ScaleExpectShape, x2ScaleExpectShape);
    auto ret = MxScaleContiguousProcess(params.x1Scale, params.transposeX1 != transposeX1Before, x1ScaleExpectShape,
                                        executor, "x1, x1Scale", apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MxScaleContiguousProcess(params.x2Scale, transposeX2View, x2ScaleExpectShape, executor, "x2, x2Scale",
                                   apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // x2 was normalized once before NZ storage validation. Do not infer transpose again.
    CHECK_RET(CheckGroupSize(params, apiName), ACLNN_ERR_PARAM_INVALID);

    ret = CheckParams(params, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return AddQuantMatmulActivationQuantTasks(params, executor);
}

} // namespace

namespace QBMMActivationQuant {

aclnnStatus GetWeightNzWorkspaceSize(const char* apiName, QuantMatmulActivationQuantParams params,
                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckWeightNzInputs(params.x1, params.x2, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    params.yDtype = params.x1->GetDataType();

    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckTensorViews(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputRanks(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(Ops::NN::CheckWeightNzViewStrideValid(params.x2), ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(PrepareInputViews(params, executorPtr), ACLNN_ERR_INNER_NULLPTR);
    const bool transposeX2Before = params.transposeX2;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::NormalizeWeightNzView(params.x2, params.transposeX2, executorPtr),
              ACLNN_ERR_INNER_NULLPTR);

    const auto weightNzShape = QuantMatmulActivationQuantAclnnCheck::GetWeightNzShape(params.x2, params.transposeX2);
    if (!QuantMatmulActivationQuantAclnnCheck::CheckWeightNzStorageShape(weightNzShape, params.x2->GetStorageShape())) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2", op::ToString(params.x2->GetStorageShape()).GetString(),
            "the storage shape of x2 must match the expected FRACTAL_NZ shape; use the WeightNZ preprocessing API "
            "to convert the input tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    ret = BuildExecutor(params, params.transposeX2 != transposeX2Before, executorPtr, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

} // namespace QBMMActivationQuant
