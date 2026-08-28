/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_quant_matmul_activation_quant.h"
#include "quant_matmul_activation_quant_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include <dlfcn.h>
#include "securec.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_matmul_activation_quant.h"
#include "quant_matmul_activation_quant_check.h"
#include "util/math_util.h"

using namespace op;
using namespace QBMMActivationQuant;
using Ops::NN::FormatString;
using Ops::NN::StripEnclosingSquareBrackets;
using Ops::NN::SwapLastTwoDimValue;

namespace {

constexpr int IDX_0 = 0;
constexpr const char* API_NAME = "aclnnQuantMatmulActivationQuantGetWorkspaceSize";

static aclnnStatus CheckFormat(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (params.x1->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x1", op::ToString(params.x1->GetStorageFormat()).GetString(),
                                                "the format of x1 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x2", op::ToString(params.x2->GetStorageFormat()).GetString(),
                                                "the format of x2 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x1Scale",
                                                op::ToString(params.x1Scale->GetStorageFormat()).GetString(),
                                                "the format of x1Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x2Scale",
                                                op::ToString(params.x2Scale->GetStorageFormat()).GetString(),
                                                "the format of x2Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.bias != nullptr && params.bias->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "bias",
                                                op::ToString(params.bias->GetStorageFormat()).GetString(),
                                                "the format of bias must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputOutDims(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    if (x1DimNum < MX_X1_DIM_MIN || x1DimNum > MX_X1_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x1", FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the shape dim of x1 must be in the range of 2 to 6"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MX_X1_DIM_MIN || x2DimNum > MX_X1_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x2", FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the shape dim of x2 must be in the range of 2 to 6"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightNdParamsDAV3510(const aclTensor* x1, const aclTensor* x2)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return ACLNN_SUCCESS;
    }

    if (x1 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "x1", "null", "x1 can not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "x2", "null", "x2 can not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    OP_LOGD("QuantMatmulWeightNd check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");
    MatmulShapeInfo shapeInfo = GetMatmulShapeInfo(params);
    CHECK_COND(CheckShapeInfoMatch(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckShapeInfoMatch failed.");

    if (!CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim, API_NAME)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMKN failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        if (shapeInfo.kDim <= 2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, "K", std::to_string(shapeInfo.kDim).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the K dimension must be greater than 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t x1InnerAxis = params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim;
        if (x1InnerAxis % 2 != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, params.transposeX1 ? "x1 M" : "x1 K", std::to_string(x1InnerAxis).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the inner axis of x1 must be even");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t x2InnerAxis = params.transposeX2 ? shapeInfo.kDim : shapeInfo.nDim;
        if (x2InnerAxis % 2 != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, params.transposeX2 ? "x2 K" : "x2 N", std::to_string(x2InnerAxis).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the inner axis of x2 must be even");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    CHECK_COND(CheckMxScaleLastDim(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckMxScaleLastDim failed.");

    if (params.bias != nullptr) {
        auto biasDimNum = params.bias->GetViewShape().GetDimNum();
        auto outDimNum = params.y->GetViewShape().GetDimNum();
        auto nDim = shapeInfo.nDim;
        if (biasDimNum != 1 && biasDimNum != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "bias", FormatString("%zuD", biasDimNum).c_str(),
                                                     "the shape dim of bias must be 1 or 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (biasDimNum == 1) {
            CHECK_COND(params.bias->GetViewShape().GetDim(0) == nDim, ACLNN_ERR_PARAM_INVALID,
                       "bias dim should be equal to N dim %ld, but is %ld", nDim,
                       params.bias->GetViewShape().GetDim(0));
        } else {
            if (outDimNum == 2 || outDimNum == 4 || outDimNum == 5 || outDimNum == 6) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    API_NAME, "bias", FormatString("%zuD", biasDimNum).c_str(),
                    FormatString("when out dim-num is %zu, bias only support 1D, but is 3D", outDimNum).c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            CHECK_COND(params.bias->GetViewShape().GetDim(1) == 1, ACLNN_ERR_PARAM_INVALID,
                       "bias 2nd dim should be 1, but is %ld", params.bias->GetViewShape().GetDim(1));
            CHECK_COND(params.bias->GetViewShape().GetDim(2) == nDim, ACLNN_ERR_PARAM_INVALID,
                       "bias 3rd dim should be equal to N dim %ld, but is %ld", nDim,
                       params.bias->GetViewShape().GetDim(2));
            int64_t inferedOutbatchValue = InferOutputShape(params);
            if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
                return ACLNN_ERR_PARAM_INVALID;
            }
            CHECK_COND(params.bias->GetViewShape().GetDim(0) == inferedOutbatchValue, ACLNN_ERR_PARAM_INVALID,
                       "bias 1st dim should be batch, but is %ld", params.bias->GetViewShape().GetDim(0));
        }
    }

    CHECK_COND(CheckExpectedShapes(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckExpectedShapes failed.");
    CHECK_COND(CheckOutputShape(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckOutputShape failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    OP_LOGD("QuantMatmulActivationQuant check params.");
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalAlg(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("QuantMatmulActivationQuant check params success.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcessOriginalShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                           const aclTensor* x2Scale)
{
    if (x1 != nullptr) {
        x1->SetOriginalShape(x1->GetViewShape());
        OP_LOGD("x1 original shape set to view shape.");
    }

    if (x2 != nullptr) {
        x2->SetOriginalShape(x2->GetViewShape());
        OP_LOGD("x2 original shape set to view shape.");
    }

    if (x1Scale != nullptr) {
        x1Scale->SetOriginalShape(x1Scale->GetViewShape());
        OP_LOGD("x1Scale original shape set to view shape.");
    }

    if (x2Scale != nullptr) {
        x2Scale->SetOriginalShape(x2Scale->GetViewShape());
        OP_LOGD("x2Scale original shape set to view shape.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSizeCommon(
    QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params, aclOpExecutor* executor)
{
    auto x2ScaleNd = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x2Scale);
    CHECK_RET(x2ScaleNd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x2Scale = x2ScaleNd;
    auto x1ScaleNd = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x1Scale);
    CHECK_RET(x1ScaleNd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x1Scale = x1ScaleNd;

    if (params.bias != nullptr) {
        auto biasNd = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.bias);
        CHECK_RET(biasNd != nullptr, ACLNN_ERR_INNER_NULLPTR);
        params.bias = biasNd;
    }

    auto reformatedX1 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x1);
    CHECK_RET(reformatedX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x1 = reformatedX1;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.x1, params.transposeX1, executor),
              ACLNN_ERR_INNER_NULLPTR);
    if (params.bias != nullptr) {
        bool biasTransposeValue = false;
        CHECK_RET(
            QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.bias, biasTransposeValue, executor),
            ACLNN_ERR_INNER_NULLPTR);
    }
    auto reformatedX2 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x2);
    CHECK_RET(reformatedX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x2 = reformatedX2;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.x2, params.transposeX2, executor),
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x1Scale, params.transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x2Scale, params.transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);

    GetTranspose(params, params.transposeX1, params.transposeX2);

    CHECK_COND(CheckGroupSize(params, API_NAME), ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed.");

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // Invoke l0 operator QuantMatmulActivationQuant for calculation.
    auto quantMatmulActivationQuantResults = l0op::QuantMatmulActivationQuant(
        params.x1, params.x2, params.bias, params.x1Scale, params.x2Scale, params.transposeX1, params.transposeX2,
        params.groupSize, params.activationType, params.y_dtype, params.quantMode, params.roundMode, params.scaleAlg,
        params.dstTypeMax, executor);

    auto yComputeOut = std::get<IDX_0>(quantMatmulActivationQuantResults);
    auto yScaleComputeOut = std::get<1>(quantMatmulActivationQuantResults);

    CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(yScaleComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, params.y, executor);
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyYScaleResult = l0op::ViewCopy(yScaleComputeOut, params.yScale, executor);
    CHECK_RET(viewCopyYScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
    const aclTensor* biasOptional, bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* yOut,
    aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulActivationQuant,
                   DFX_IN(x1, x2, x1ScaleOptional, x2Scale, biasOptional, transposeX1, transposeX2, groupSize,
                          activationType, quantMode, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(yOut, yScaleOut));

    auto ret = CheckWeightNdParamsDAV3510(x1, x2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto y_dtype = x1->GetDataType();
    QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams params{
        x1,          x2,        x1ScaleOptional, x2Scale, biasOptional, yOut,      yScaleOut, transposeX1,
        transposeX2, groupSize, activationType,  y_dtype, quantMode,    roundMode, scaleAlg,  dstTypeMax};

    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // 空tensor 处理
    if (params.x1->IsEmpty() || params.x2->IsEmpty() || (params.x1Scale != nullptr && params.x1Scale->IsEmpty()) ||
        (params.x2Scale != nullptr && params.x2Scale->IsEmpty()) ||
        (params.bias != nullptr && params.bias->IsEmpty()) || params.y->IsEmpty() || params.yScale->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            API_NAME, "x1, x2, x1Scale, x2Scale, bias, y, yScale",
            Ops::NN::FormatString(
                "%s, %s, %s, %s, %s, %s, %s", op::ToString(x1->GetViewShape()).GetString(),
                op::ToString(x2->GetViewShape()).GetString(),
                params.x1Scale != nullptr ? op::ToString(params.x1Scale->GetViewShape()).GetString() : "null",
                op::ToString(x2Scale->GetViewShape()).GetString(),
                params.bias != nullptr ? op::ToString(params.bias->GetViewShape()).GetString() : "null",
                op::ToString(params.y->GetViewShape()).GetString(),
                op::ToString(params.yScale->GetViewShape()).GetString())
                .c_str(),
            Ops::NN::FormatString("The shapes of %s cannot be %s", "x1, x2, x1Scale, x2Scale, bias, y, yScale", "empty")
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Step 1: 设置original_shape（必须在Contiguous之前）
    ret = PreProcessOriginalShape(params.x1, params.x2, params.x1Scale, params.x2Scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ret = aclnnQuantMatmulActivationQuantGetWorkspaceSizeCommon(params, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Standard syntax, get the size of workspace needed during computation.
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulActivationQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulActivationQuant);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantMatmulActivationQuant launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
