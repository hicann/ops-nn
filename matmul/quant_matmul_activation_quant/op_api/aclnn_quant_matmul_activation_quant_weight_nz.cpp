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
#include "aclnn_quant_matmul_activation_quant_weight_nz.h"
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
#include "quant_matmul_activation_quant_checker.h"
#include "util/math_util.h"

using namespace op;
using namespace QBMMActivationQuant;
using Ops::NN::FormatString;
using Ops::NN::SwapLastTwoDimValue;

namespace {

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr const char* API_NAME = "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize";

static aclnnStatus CheckFormat(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (params.x1->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x1", op::ToString(params.x1->GetStorageFormat()).GetString(),
                                                "the format of x1 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(API_NAME, "x2", op::ToString(params.x2->GetStorageFormat()).GetString(),
                                                "the format of x2 must be FORMAT_FRACTAL_NZ");
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
    auto x2DimNum = params.x2->GetStorageShape().GetDimNum();
    if (x1DimNum < MX_X1_DIM_MIN || x1DimNum > MX_X1_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x1", FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the rank of x1 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MX_X2_DIM_MIN || x2DimNum > MX_X2_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x2", FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the rank of x2 must be in the range [4, 8]"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightNzParamsDAV3510(const aclTensor* x1, const aclTensor* x2)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        SocVersion socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "SOC version %s is not supported", op::ToString(socVersion).GetString());
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    if (x1 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "x1", "null", "x1 cannot be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "x2", "null", "x2 cannot be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(API_NAME, "x2", op::ToString(x2->GetStorageFormat()).GetString(),
                                               "the format of x2 must be FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (x2->GetDataType() == op::DataType::DT_FLOAT8_E5M2) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "x2", op::ToString(x2->GetDataType()).GetString(),
                                              "the FLOAT8_E5M2 dtype of x2 is supported only in ND format");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("WeightNZ input validation succeeded.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    CHECK_RET(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    MatmulShapeInfo shapeInfo = GetMatmulShapeInfo(params);
    CHECK_RET(CheckShapeInfoMatch(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // NZ情况下，x2的k和n不能为1
    int64_t dim1 = params.x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = params.x2->GetViewShape().GetDimNum() - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM;
    if (params.x2->GetViewShape().GetDim(dim2) == 1 || params.x2->GetViewShape().GetDim(dim1) == 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            API_NAME, params.transposeX2 ? "x2 N, x2 K" : "x2 K, x2 N",
            FormatString("%lld, %lld", static_cast<long long>(params.x2->GetViewShape().GetDim(dim2)),
                         static_cast<long long>(params.x2->GetViewShape().GetDim(dim1)))
                .c_str(),
            "when the format of x2 is FRACTAL_NZ, the K and N dimensions of x2 cannot be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim, API_NAME), ACLNN_ERR_PARAM_INVALID);
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        if (params.transposeX1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, "transposeX1", "true",
                "when the format of x2 is FRACTAL_NZ and the dtypes of x1, x2 and y are FP4, x1 cannot be transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (shapeInfo.kDim <= 2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, "K", std::to_string(shapeInfo.kDim).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the K dimension must be greater than 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t x1InnerAxis = params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim;
        if (x1InnerAxis % FP4_PACK_RATIO != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, params.transposeX1 ? "x1 M" : "x1 K", std::to_string(x1InnerAxis).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the inner axis of x1 must be even");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t x2InnerAxis = params.transposeX2 ? shapeInfo.kDim : shapeInfo.nDim;
        if (x2InnerAxis % FP4_PACK_RATIO != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, params.transposeX2 ? "x2 K" : "x2 N", std::to_string(x2InnerAxis).c_str(),
                "when the dtypes of x1, x2 and y are FP4, the inner axis of x2 must be even");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    CHECK_RET(CheckMxScaleLastDim(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (params.bias != nullptr) {
        auto biasDimNum = params.bias->GetViewShape().GetDimNum();
        auto outDimNum = params.y->GetViewShape().GetDimNum();
        auto nDim = shapeInfo.nDim;
        if (biasDimNum != 1 && biasDimNum != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "bias", FormatString("%zuD", biasDimNum).c_str(),
                                                     "the rank of bias must be 1 or 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (biasDimNum == 1) {
            if (params.bias->GetViewShape().GetDim(0) != nDim) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    API_NAME, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                    FormatString("the shape of bias must be [%lld]", static_cast<long long>(nDim)).c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (outDimNum == 2 || outDimNum == 4 || outDimNum == 5 || outDimNum == 6) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    API_NAME, "bias", FormatString("%zuD", biasDimNum).c_str(),
                    FormatString("when output rank is %zu, only 1D bias is supported, but got 3D", outDimNum).c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (params.bias->GetViewShape().GetDim(1) != 1) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(API_NAME, "bias",
                                                      op::ToString(params.bias->GetViewShape()).GetString(),
                                                      "the 2nd dimension of bias must be 1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (params.bias->GetViewShape().GetDim(2) != nDim) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    API_NAME, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                    FormatString("the 3rd dimension of bias must be %lld", static_cast<long long>(nDim)).c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            int64_t inferedOutbatchValue = InferOutputShape(params, API_NAME);
            if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (params.bias->GetViewShape().GetDim(0) != inferedOutbatchValue) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    API_NAME, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                    FormatString("the 1st dimension of bias must be %lld", static_cast<long long>(inferedOutbatchValue))
                        .c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    CHECK_RET(CheckExpectedShapes(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputShape(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    OP_LOGD("Parameter validation started.");
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalAlg(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Parameter validation succeeded.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcessOriginalShape(const aclTensor* x1, const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    // original shape must be set before contiguous
    if (x1 != nullptr) {
        x1->SetOriginalShape(x1->GetViewShape());
        OP_LOGD("Set x1 original shape to its view shape.");
    }

    if (x1Scale != nullptr) {
        x1Scale->SetOriginalShape(x1Scale->GetViewShape());
        OP_LOGD("Set x1Scale original shape to its view shape.");
    }

    if (x2Scale != nullptr) {
        x2Scale->SetOriginalShape(x2Scale->GetViewShape());
        OP_LOGD("Set x2Scale original shape to its view shape.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSizeCommon(
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
    CHECK_RET(MxScaleContiguousProcess(params.x1Scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x2Scale, executor), ACLNN_ERR_INNER_NULLPTR);

    // 设置x2的OriginalShape为它的ViewShape
    auto retNZProcess = QuantMatmulActivationQuantAclnnCheck::WeightNZCaseProcess(params.x2, params.transposeX2,
                                                                                  executor);
    CHECK_RET(retNZProcess == ACLNN_SUCCESS, retNZProcess);

    GetTranspose(params, params.transposeX1, params.transposeX2);

    CHECK_RET(CheckGroupSize(params, API_NAME), ACLNN_ERR_PARAM_INVALID);

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    // Invoke l0 operator QuantMatmulActivationQuant for calculation.
    auto quantMatmulActivationQuantResults = l0op::QuantMatmulActivationQuant(
        params.x1, params.x2, params.bias, params.x1Scale, params.x2Scale, params.transposeX1, params.transposeX2,
        params.groupSize, params.activationType, params.y_dtype, params.quantMode, params.roundMode, params.scaleAlg,
        params.dstTypeMax, executor);

    auto yComputeOut = std::get<IDX_0>(quantMatmulActivationQuantResults);
    auto yScaleComputeOut = std::get<IDX_1>(quantMatmulActivationQuantResults);

    // 校验输出不为空
    CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(yScaleComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果拷贝到输出tensor
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
aclnnStatus aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
    const aclTensor* biasOptional, bool transposeX1, bool transposeX2, int64_t groupSize, char* activationType,
    char* quantMode, char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* y, aclTensor* yScale,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulActivationQuantWeightNz,
                   DFX_IN(x1, x2, x1ScaleOptional, x2Scale, biasOptional, transposeX1, transposeX2, groupSize,
                          activationType, quantMode, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(y, yScale));

    auto ret = CheckWeightNzParamsDAV3510(x1, x2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto y_dtype = x1->GetDataType();
    QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams params{
        x1,          x2,        x1ScaleOptional, x2Scale, biasOptional, y,         yScale,   transposeX1,
        transposeX2, groupSize, activationType,  y_dtype, quantMode,    roundMode, scaleAlg, dstTypeMax};

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

    // 在 Contiguous 之前保留输入的 original_shape。
    ret = PreProcessOriginalShape(params.x1, params.x1Scale, params.x2Scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    params.transposeX2 = GetTransposeAttrValue(x2, transposeX2, false);

    op::Shape weightNzShape = QuantMatmulActivationQuantAclnnCheck::GetWeightNzShape(x2, params.transposeX2);
    if (!QuantMatmulActivationQuantAclnnCheck::CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            API_NAME, "x2", op::ToString(x2->GetStorageShape()).GetString(),
            "the storage shape of x2 must match the expected FRACTAL_NZ shape; use the WeightNZ preprocessing API "
            "to convert the input tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    x2 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNZFormat(x2, weightNzShape, executorPtr);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    params.x2 = x2;

    ret = aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSizeCommon(params, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Standard syntax, get the size of workspace needed during computation.
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulActivationQuantWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulActivationQuantWeightNz);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantMatmulActivationQuantWeightNz launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
