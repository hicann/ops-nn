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
#include "quant_matmul_activation_quant_check.h"
#include "util/math_util.h"

using namespace op;
using namespace QBMMActivationQuant;
using Ops::NN::FormatString;
using Ops::NN::StripEnclosingSquareBrackets;
using Ops::NN::SwapLastTwoDimValue;

namespace {
struct MatmulShapeInfo {
    int64_t mDim;
    int64_t kDim;
    int64_t nDim;
};

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;

static aclnnStatus CheckNotNull(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    OP_CHECK_NULL(params.x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x1Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.y, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.yScale, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (params.x1->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1",
                                                op::ToString(params.x1->GetStorageFormat()).GetString(),
                                                "the format of x1 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2",
                                                op::ToString(params.x2->GetStorageFormat()).GetString(),
                                                "the format of x2 must be FORMAT_FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1Scale",
                                                op::ToString(params.x1Scale->GetStorageFormat()).GetString(),
                                                "the format of x1Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2Scale",
                                                op::ToString(params.x2Scale->GetStorageFormat()).GetString(),
                                                "the format of x2Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.bias != nullptr && params.bias->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "bias",
                                                op::ToString(params.bias->GetStorageFormat()).GetString(),
                                                "the format of bias must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsMxQuantDim(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    // scale 由固定 3 维（M/K/2 或 K/N/2）加上与 x1 一致的 batch 维度组成
    int64_t x1DimNum = static_cast<int64_t>(params.x1->GetViewShape().GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MX_X1_DIM), 0);
    int64_t expectedScaleDimNum = static_cast<int64_t>(MX_X1_SCALE_DIM) + x1BatchDimNum;

    auto x1ScaleDimNum = params.x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    if (static_cast<int64_t>(x2ScaleDimNum) != expectedScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2Scale",
            FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x2Scale must be %ld "
                         "(batch dim of x1 %ld + fixed dim %zu)",
                         expectedScaleDimNum, x1BatchDimNum, MX_X2_SCALE_DIM)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (static_cast<int64_t>(x1ScaleDimNum) != expectedScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1Scale",
            FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x1Scale must be %ld "
                         "(batch dim of x1 %ld + fixed dim %zu)",
                         expectedScaleDimNum, x1BatchDimNum, MX_X1_SCALE_DIM)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputDtypeValid(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (!CheckType(params.x1->GetDataType(), X1_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1",
                                              op::ToString(params.x1->GetDataType()).GetString(),
                                              FormatString("the dtype of x1 must be in dtype support list %s",
                                                           op::ToString(X1_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.x2->GetDataType(), X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2",
                                              op::ToString(params.x2->GetDataType()).GetString(),
                                              FormatString("the dtype of x2 must be in dtype support list %s",
                                                           op::ToString(X2_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxfp8DtypeValid(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (CheckInputDtypeValid(params) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1Scale",
            op::ToString(params.x1Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x1Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2Scale",
            op::ToString(params.x2Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x2Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("QuantMatmulActivationQuant CheckMxfp8DtypeValid success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    auto x1ScaleDtype = params.x1Scale->GetDataType();
    auto x2ScaleDtype = params.x2Scale->GetDataType();
    auto yDtype = params.y->GetDataType();
    auto yScaleDtype = params.yScale->GetDataType();
    if ((x1Dtype == DataType::DT_FLOAT8_E4M3FN || x1Dtype == DataType::DT_FLOAT8_E5M2) &&
        x2Dtype == DataType::DT_FLOAT8_E4M3FN &&
        (yDtype == DataType::DT_FLOAT8_E4M3FN || yDtype == DataType::DT_FLOAT8_E5M2) &&
        yScaleDtype == DataType::DT_FLOAT8_E8M0) {
        CHECK_COND(IsMxQuantDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check IsMxQuantDim failed.");
        if (params.bias != nullptr && params.bias->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "bias",
                                                  op::ToString(params.bias->GetDataType()).GetString(),
                                                  "the dtype of bias must be FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return CheckMxfp8DtypeValid(params);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1, x2, x1Scale, x2Scale, y, yScale",
            FormatString("%s, %s, %s, %s", op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                         op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString())
                .c_str(),
            FormatString("when the dtypes of x1 and x2 are %s and %s, and the dtypes of x1Scale and x2Scale are %s "
                         "and %s, this dtype combination can not be supported",
                         op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                         op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString())
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
}

static aclnnStatus CheckOptionalAlg(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    CHECK_RET(params.activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const std::string activationType(params.activationType);
    if (activationType != "gelu_tanh" && activationType != "gelu_erf") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize",
                                              "activationType", activationType,
                                              "The activationType must be gelu_tanh or gelu_erf");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(params.quantMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const std::string quantMode(params.quantMode);
    if (quantMode != "mx") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "quantMode",
                                              quantMode, "The quantMode must be mx");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(params.roundMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    std::string roundMode(params.roundMode);
    if (roundMode != "rint" && roundMode != "floor" && roundMode != "round") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "roundMode",
                                              roundMode,
                                              "roundMode optional values are rint/floor/round, it's enabled when "
                                              "dynamic mx quant, fp8 only support rint, fp4 support all");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.scaleAlg != 0 && params.scaleAlg != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "scaleAlg",
                                              std::to_string(params.scaleAlg), "The scaleAlg optional values are 0/1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckType(params.y->GetDataType(), Y_DTYPE_SUPPORT_LIST) && roundMode != "rint") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "roundMode",
                                              roundMode,
                                              FormatString("roundMode must be rint when the dtype of y in %s",
                                                           op::ToString(Y_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.y->GetDataType(), Y_DTYPE_SUPPORT_LIST) && params.scaleAlg == 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "scaleAlg",
                                              std::to_string(params.scaleAlg),
                                              FormatString("scaleAlg can't be 1 when the dtype of y not in %s",
                                                           op::ToString(Y_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static MatmulShapeInfo GetMatmulShapeInfo(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    int64_t x1DimNum = params.x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    return {
        params.transposeX1 ?
            params.x1->GetViewShape().GetDim(x1DimNum - 1) :
            params.x1->GetViewShape().GetDim(x1DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM),
        params.transposeX1 ?
            params.x1->GetViewShape().GetDim(x1DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM) :
            params.x1->GetViewShape().GetDim(x1DimNum - 1),
        params.transposeX2 ?
            params.x2->GetViewShape().GetDim(x2DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM) :
            params.x2->GetViewShape().GetDim(x2DimNum - 1),
    };
}

static aclnnStatus CheckInputOutDims(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetStorageShape().GetDimNum();
    if (x1DimNum < MX_X1_DIM_MIN || x1DimNum > MX_X1_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1",
                                                 FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the shape dim of x1 must be in the range of 2 to 6"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MX_X2_DIM_MIN || x2DimNum > MX_X2_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2",
                                                 FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the shape dim of x2 must be in the range of 4 to 8"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapeInfoMatch(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params,
                                       const MatmulShapeInfo& shapeInfo)
{
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    int64_t x2KDim = params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - 1) :
                                          params.x2->GetViewShape().GetDim(
                                              x2DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM);
    if (shapeInfo.kDim != x2KDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1 K, x2 K",
                                               FormatString("%ld, %ld", shapeInfo.kDim, x2KDim).c_str(),
                                               "the K dimension of x1 and x2 must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool CheckMKN(int64_t m, int64_t k, int64_t n)
{
    if (m <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1 M",
                                              std::to_string(m).c_str(), "the M dimension of x1 must be positive");
        return false;
    }
    if (k <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "K",
                                              std::to_string(k).c_str(),
                                              "the K dimension of x1 and x2 must be positive");
        return false;
    }
    if (n <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2 N",
                                              std::to_string(n).c_str(), "the N dimension of x2 must be positive");
        return false;
    }
    return true;
}

static aclnnStatus CheckWeightNzParamsDAV3510(const aclTensor* x1, const aclTensor* x2)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        SocVersion socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    if (x1 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1", "null",
                                              "x1 can not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2", "null",
                                              "x2 can not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2",
                                               op::ToString(x2->GetStorageFormat()).GetString(),
                                               "the format of x2 must be FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // NZ情况下，x2的k和n不能为1
    int64_t dim1 = x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = x2->GetViewShape().GetDimNum() - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM;
    if (x2->GetViewShape().GetDim(dim2) == 1 || x2->GetViewShape().GetDim(dim1) == 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2 K, x2 N",
            FormatString("%ld, %ld", x2->GetViewShape().GetDim(dim2), x2->GetViewShape().GetDim(dim1)).c_str(),
            "when the format of x2 is FRACTAL_NZ, the k dimension and n dimension of x2 can not be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("QuantMatmulWeightNz check params success.");
    return ACLNN_SUCCESS;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static aclnnStatus CheckMxScaleLastDim(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    if (!IsMicroScaling(params.x1Scale, params.x2Scale)) {
        return ACLNN_SUCCESS;
    }

    auto scale1LastDimValue = params.x1Scale->GetViewShape().GetDim(params.x1Scale->GetViewShape().GetDimNum() - 1);
    auto scale2LastDimValue = params.x2Scale->GetViewShape().GetDim(params.x2Scale->GetViewShape().GetDimNum() - 1);
    if (scale1LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1Scale->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x1Scale must be %d",
                         MXFP_MULTI_BASE_SIZE)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale2LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x2Scale must be %d",
                         MXFP_MULTI_BASE_SIZE)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static void GetExpectedScaleShape(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params,
                                  const MatmulShapeInfo& shapeInfo, op::Shape& x1ScaleExpectShape,
                                  op::Shape& x2ScaleExpectShape)
{
    if (!IsMicroScaling(params.x1Scale, params.x2Scale)) {
        x1ScaleExpectShape = {1};
        x2ScaleExpectShape = {1};
        return;
    }

    // batch 维度与 x1 保持一致
    const auto& x1View = params.x1->GetViewShape();
    int64_t x1DimNum = static_cast<int64_t>(x1View.GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MX_X1_DIM), 0);

    x1ScaleExpectShape = op::Shape();
    x2ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < x1BatchDimNum; ++i) {
        x1ScaleExpectShape.AppendDim(x1View.GetDim(i));
        x2ScaleExpectShape.AppendDim(x1View.GetDim(i));
    }
    if (params.transposeX1) {
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
    } else {
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
    }
    x1ScaleExpectShape.AppendDim(MXFP_MULTI_BASE_SIZE);

    if (params.transposeX2) {
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
    } else {
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
    }
    x2ScaleExpectShape.AppendDim(MXFP_MULTI_BASE_SIZE);
}

static aclnnStatus CheckExpectedShapes(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params,
                                       const MatmulShapeInfo& shapeInfo)
{
    auto& x1View = params.x1->GetViewShape();
    auto& x2View = params.x2->GetViewShape();
    int64_t x1DimNum = x1View.GetDimNum();
    int64_t x2DimNum = x2View.GetDimNum();

    // 维度下限校验：x1/x2 此处均为 viewShape（ND 逻辑维度），下限与 x1 一致
    if (x1DimNum < MX_X1_DIM || x2DimNum < MX_X1_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "unsupported dim combination: x1DimNum=%ld, x2DimNum=%ld", x1DimNum, x2DimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 1. batch 维 broadcast 合法性校验（2D 时 batchDimNum=0，循环不执行）
    int64_t x1BatchCount = x1DimNum - 2;
    int64_t x2BatchCount = x2DimNum - 2;
    int64_t batchDimNum = std::max(x1BatchCount, x2BatchCount);
    for (int64_t i = 0; i < batchDimNum; ++i) {
        // 右对齐：batch 维从左向右编号，不足的补 1
        int64_t x1Idx = i - (batchDimNum - x1BatchCount);
        int64_t x2Idx = i - (batchDimNum - x2BatchCount);
        int64_t x1BatchDim = (x1Idx >= 0) ? x1View.GetDim(x1Idx) : 1;
        int64_t x2BatchDim = (x2Idx >= 0) ? x2View.GetDim(x2Idx) : 1;
        if (x1BatchDim != x2BatchDim && x1BatchDim != 1 && x2BatchDim != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch dim %ld mismatch: x1=%ld, x2=%ld", i, x1BatchDim, x2BatchDim);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 2. 校验 x1 最后两维
    int64_t x1M = params.transposeX1 ? x1View.GetDim(x1DimNum - 1) : x1View.GetDim(x1DimNum - 2);
    int64_t x1K = params.transposeX1 ? x1View.GetDim(x1DimNum - 2) : x1View.GetDim(x1DimNum - 1);
    if (x1M != shapeInfo.mDim || x1K != shapeInfo.kDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1",
                                              StripEnclosingSquareBrackets(op::ToString(x1View).GetString()).c_str(),
                                              FormatString("x1 last two dims must be [%ld, %ld], but got [%ld, %ld]",
                                                           params.transposeX1 ? shapeInfo.kDim : shapeInfo.mDim,
                                                           params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim,
                                                           x1View.GetDim(x1DimNum - 2), x1View.GetDim(x1DimNum - 1))
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 校验 x2 最后两维
    int64_t x2K = params.transposeX2 ? x2View.GetDim(x2DimNum - 1) : x2View.GetDim(x2DimNum - 2);
    int64_t x2N = params.transposeX2 ? x2View.GetDim(x2DimNum - 2) : x2View.GetDim(x2DimNum - 1);
    if (x2K != shapeInfo.kDim || x2N != shapeInfo.nDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2",
                                              StripEnclosingSquareBrackets(op::ToString(x2View).GetString()).c_str(),
                                              FormatString("x2 last two dims must be [%ld, %ld], but got [%ld, %ld]",
                                                           params.transposeX2 ? shapeInfo.nDim : shapeInfo.kDim,
                                                           params.transposeX2 ? shapeInfo.kDim : shapeInfo.nDim,
                                                           x2View.GetDim(x2DimNum - 2), x2View.GetDim(x2DimNum - 1))
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShape(params, shapeInfo, x1ScaleExpectShape, x2ScaleExpectShape);

    if (params.x1Scale->GetViewShape() != x1ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1Scale->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x1Scale must be %s", op::ToString(x1ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetViewShape() != x2ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x2Scale must be %s", op::ToString(x2ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static int64_t InferOutputShape(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    int64_t inferedOutbatchValue = 1;
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    auto& longShapeTensor = x1DimNum > x2DimNum ? params.x1 : params.x2;
    auto& shortShapeTensor = x1DimNum > x2DimNum ? params.x2 : params.x1;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i + QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM < outDimNum; i++) {
        auto shortDimValue = i < validOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - validOffset);
        auto longDimValue = longShapeTensor->GetViewShape().GetDim(i);
        if (shortDimValue > 1 && longDimValue > 1 && shortDimValue != longDimValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Current short dim value %ld and long dim value %ld are not supported for broadcasting.",
                    shortDimValue, longDimValue);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
    }
    return inferedOutbatchValue;
}

static aclnnStatus CheckShape(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");
    MatmulShapeInfo shapeInfo = GetMatmulShapeInfo(params);
    CHECK_COND(CheckShapeInfoMatch(params, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckShapeInfoMatch failed.");

    if (!CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMKN failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_COND(CheckMxScaleLastDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckMxScaleLastDim failed.");

    if (params.bias != nullptr) {
        auto biasDimNum = params.bias->GetViewShape().GetDimNum();
        auto outDimNum = params.y->GetViewShape().GetDimNum();
        auto nDim = shapeInfo.nDim;
        if (biasDimNum != 1 && biasDimNum != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "bias",
                                                     FormatString("%zuD", biasDimNum).c_str(),
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
                    "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "bias",
                    FormatString("%zuD", biasDimNum).c_str(),
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

    return CheckExpectedShapes(params, shapeInfo);
}

static aclnnStatus CheckParams(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    OP_LOGD("QuantMatmulActivationQuant check params.");
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalAlg(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("QuantMatmulActivationQuant check params success.");

    return ACLNN_SUCCESS;
}
static aclnnStatus PreProcessOriginalShape(const aclTensor* x1, const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    // original shape must be set before contiguous
    if (x1 != nullptr) {
        x1->SetOriginalShape(x1->GetViewShape());
        OP_LOGD("x1 original shape set to view shape.");
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

static inline bool MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, bool transpose, aclOpExecutor* executor)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum -
                                                                 QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM);
    // 3: 倒数第3维
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&
        mxScaleTensor->GetViewStrides()[dimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM] ==
            lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4: batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (transpose) {
            if (lastSecondDim == 1 && lastThirdDim == 1) {
                transposeFlag = false;
            }
        } else {
            if (lastSecondDim == 1 || lastThirdDim == 1) {
                transposeFlag = false;
            }
        }
    }

    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM, lastThirdDim);
        // 3: 倒数第3维
        swapedShape.SetDim(dimNum - 3, lastSecondDim);
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim)
{
    if ((tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim)) &&
        (tensor->GetViewShape().GetDim(secondLastDim) == 1)) {
        OP_LOGD("QuantMatmulActivationQuant special case, no need to set transpose attr value.");
        return true;
    }
    return false;
}

static bool GetTransposeAttrValue(const aclTensor* tensor, bool transpose, bool checkSpecialCase = true)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 &&
        (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantMatmulActivationQuant GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if (!checkSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static void GetTranspose(const QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params, bool& transposeX1,
                         bool& transposeX2)
{
    transposeX1 = GetTransposeAttrValue(params.x1, transposeX1, true);
    transposeX2 = GetTransposeAttrValue(params.x2, transposeX2, true);
    OP_LOGD("QuantMatmulActivationQuant attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

static bool CheckGroupSize(QBMMActivationQuant::QuantMatmulActivationQuantWeightNzParams& params)
{
    auto groupSize = params.groupSize;
    if (groupSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "groupSize",
                                              std::to_string(groupSize).c_str(), "groupSize can not be negative");
        return false;
    }
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;

    if (groupSizeK == 0 && groupSizeM == 0 && groupSizeN == 0) {
        params.groupSize = (1UL << GROUP_M_OFFSET) | (1UL << GROUP_N_OFFSET) |
                           static_cast<uint64_t>(PERGROUP_GROUP_SIZE);
    } else if (groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1UL || groupSizeN != 1UL) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%ld, %lu, %lu, %lu", groupSize, groupSizeM, groupSizeN, groupSizeK).c_str(),
            "when the quantization mode is mx, groupSize must be 4295032864 and Torch API group_sizes must be [1, "
            "1, 32]");
        return false;
    }

    OP_LOGD("QuantMatmulActivationQuant check group_size success.");
    return true;
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
    CHECK_RET(MxScaleContiguousProcess(params.x1Scale, params.transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x2Scale, params.transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);

    // 设置x2的OriginalShape为它的ViewShape
    auto retNZProcess = QuantMatmulActivationQuantAclnnCheck::WeightNZCaseProcess(params.x2, params.transposeX2,
                                                                                  executor);
    CHECK_RET(retNZProcess == ACLNN_SUCCESS, retNZProcess);

    GetTranspose(params, params.transposeX1, params.transposeX2);

    CHECK_COND(CheckGroupSize(params), ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed.");

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
    const aclTensor* biasOptional, bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* y, aclTensor* yScale,
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
        (params.bias != nullptr && params.bias->IsEmpty())) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnQuantMatmulActivationQuantWeightNzGetWorkspaceSize", "x1, x2, x1Scale, x2Scale, bias",
            Ops::NN::FormatString(
                "%s, %s, %s, %s, %s", op::ToString(x1->GetViewShape()).GetString(),
                op::ToString(x2->GetViewShape()).GetString(), op::ToString(x1ScaleOptional->GetViewShape()).GetString(),
                op::ToString(x2Scale->GetViewShape()).GetString(),
                params.bias != nullptr ? op::ToString(params.bias->GetViewShape()).GetString() : "null")
                .c_str(),
            Ops::NN::FormatString("The shapes of %s cannot be %s", "x1, x2, x1Scale, x2Scale, bias", "empty").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Step 1: 设置original_shape（必须在Contiguous之前）
    ret = PreProcessOriginalShape(params.x1, params.x1Scale, params.x2Scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");

    CHECK_RET(x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    params.transposeX2 = GetTransposeAttrValue(x2, transposeX2, false);

    op::Shape weightNzShape = QuantMatmulActivationQuantAclnnCheck::GetWeightNzShape(x2, params.transposeX2);
    if (!QuantMatmulActivationQuantAclnnCheck::CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2'format only support NZ, but now x2's format is not NZ(Ascend affinity format). \
            aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be used to convert the input format from ND to Ascend \
            affinity format.");
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
