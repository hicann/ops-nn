/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_UTIL_H
#define OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_UTIL_H
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/log_format_util.h"
#include "aclnn_kernels/contiguous.h"
#include "quant_matmul_activation_quant_check.h"
#include "log/log.h"

namespace QBMMActivationQuant {
using namespace op;
using Ops::NN::FormatString;
using Ops::NN::StripEnclosingSquareBrackets;
using Ops::NN::SwapLastTwoDimValue;
struct QuantMatmulActivationQuantWeightNzParams {
    const aclTensor* x1 = nullptr;
    const aclTensor* x2 = nullptr;
    const aclTensor* x1Scale = nullptr;
    const aclTensor* x2Scale = nullptr;
    const aclTensor* bias = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    const char* activationType;
    int64_t y_dtype;
    const char* quantMode;
    const char* roundMode;
    int64_t scaleAlg;
    double dstTypeMax;
};

static const std::initializer_list<op::DataType> X1_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};

constexpr uint32_t MX_X1_DIM = 2U;
constexpr uint32_t MX_X1_DIM_MIN = 2U;
constexpr uint32_t MX_X1_DIM_MAX = 6L;
constexpr uint32_t MX_X2_DIM = 2U;
constexpr uint32_t MX_X2_DIM_MIN = 4U;
constexpr uint32_t MX_X2_DIM_MAX = 8L;
constexpr uint32_t MX_X1_SCALE_DIM = 3U;
constexpr uint32_t MX_X2_SCALE_DIM = 3U;
constexpr uint32_t PERTENSOR_SCALE_DIM = 1U;
constexpr uint32_t Y_INPUT_DIM = 2U;
constexpr uint32_t Y_OUTPUT_DIM = 2U;
constexpr uint32_t MX_X1_PER_TOKEN_SCALE_DIM = 3U;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr int64_t SPLIT_SIZE = 64L;
static constexpr int PENULTIMATE_DIM = 2;

static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const int64_t PERGROUP_GROUP_SIZE = 32L;
static const size_t MX_SCALE_MAX_DIM = 3;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;

static inline bool isA8W4Float(const aclTensor* x1, const aclTensor* x2)
{
    if (x1 == nullptr || x2 == nullptr) {
        return false;
    }
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
           (x2->GetDataType() == op::DataType::DT_FLOAT || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1);
}

static inline bool IsFloatEqual(float a, float b) { return std::abs(a - b) <= std::numeric_limits<float>::epsilon(); }

struct MatmulShapeInfo {
    int64_t mDim;
    int64_t kDim;
    int64_t nDim;
};

static inline aclnnStatus CheckNotNull(const QuantMatmulActivationQuantWeightNzParams& params)
{
    OP_CHECK_NULL(params.x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x1Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.y, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.yScale, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool IsMxFp8Input(const aclTensor* x1, const aclTensor* x2, const aclTensor* y, const aclTensor* yScale)
{
    if (x1 == nullptr || x2 == nullptr || y == nullptr || yScale == nullptr) {
        return false;
    }
    auto x1Dtype = x1->GetDataType();
    auto x2Dtype = x2->GetDataType();
    auto yDtype = y->GetDataType();
    if (!(x1Dtype == op::DataType::DT_FLOAT8_E4M3FN || x1Dtype == op::DataType::DT_FLOAT8_E5M2)) {
        return false;
    }
    if (!(x2Dtype == op::DataType::DT_FLOAT8_E4M3FN || x2Dtype == op::DataType::DT_FLOAT8_E5M2)) {
        return false;
    }
    if (yDtype != op::DataType::DT_FLOAT8_E4M3FN && yDtype != op::DataType::DT_FLOAT8_E5M2) {
        return false;
    }
    if (yScale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        return false;
    }
    return true;
}

static inline bool IsMxFp4Input(const aclTensor* x1, const aclTensor* x2, const aclTensor* y, const aclTensor* yScale)
{
    if (x1 == nullptr || x2 == nullptr || y == nullptr || yScale == nullptr) {
        return false;
    }
    return x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
           y->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && yScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim)
{
    if ((tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim)) &&
        (tensor->GetViewShape().GetDim(secondLastDim) == 1)) {
        OP_LOGD("QuantMatmulActivationQuant special case, no need to set transpose attr value.");
        return true;
    }
    return false;
}

static inline bool GetTransposeAttrValue(const aclTensor* tensor, bool transpose, bool checkSpecialCase = true)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 &&
        (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantMatmulActivationQuant GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        if (!checkSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static inline void GetTranspose(const QuantMatmulActivationQuantWeightNzParams& params, bool& transposeX1,
                                bool& transposeX2)
{
    transposeX1 = GetTransposeAttrValue(params.x1, transposeX1, true);
    transposeX2 = GetTransposeAttrValue(params.x2, transposeX2, true);
    OP_LOGD("QuantMatmulActivationQuant attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

static inline MatmulShapeInfo GetMatmulShapeInfo(const QuantMatmulActivationQuantWeightNzParams& params)
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

static inline void GetExpectedScaleShape(const QuantMatmulActivationQuantWeightNzParams& params,
                                         const MatmulShapeInfo& shapeInfo, op::Shape& x1ScaleExpectShape,
                                         op::Shape& x2ScaleExpectShape)
{
    if (!IsMicroScaling(params.x1Scale, params.x2Scale)) {
        x1ScaleExpectShape = {1};
        x2ScaleExpectShape = {1};
        return;
    }

    const auto& x1View = params.x1->GetViewShape();
    const auto& x2View = params.x2->GetViewShape();
    int64_t x1DimNum = static_cast<int64_t>(x1View.GetDimNum());
    int64_t x2DimNum = static_cast<int64_t>(x2View.GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MX_X1_DIM), 0);
    int64_t x2BatchDimNum = std::max<int64_t>(x2DimNum - static_cast<int64_t>(MX_X2_DIM), 0);
    int64_t maxBatchDimNum = std::max(x1BatchDimNum, x2BatchDimNum);

    x1ScaleExpectShape = op::Shape();
    x2ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < maxBatchDimNum; ++i) {
        int64_t x1Idx = i - (maxBatchDimNum - x1BatchDimNum);
        int64_t x2Idx = i - (maxBatchDimNum - x2BatchDimNum);
        int64_t x1BatchDim = (x1Idx >= 0) ? x1View.GetDim(x1Idx) : 1;
        int64_t x2BatchDim = (x2Idx >= 0) ? x2View.GetDim(x2Idx) : 1;
        x1ScaleExpectShape.AppendDim(x1BatchDim);
        x2ScaleExpectShape.AppendDim(x2BatchDim);
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

static inline int64_t InferOutputShape(const QuantMatmulActivationQuantWeightNzParams& params)
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

static inline aclnnStatus IsMxQuantDim(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    int64_t x1DimNum = static_cast<int64_t>(params.x1->GetViewShape().GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MX_X1_DIM), 0);
    int64_t expectedScaleDimNum = static_cast<int64_t>(MX_X1_SCALE_DIM) + x1BatchDimNum;

    auto x1ScaleDimNum = params.x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    if (static_cast<int64_t>(x2ScaleDimNum) != expectedScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x2Scale", FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x2Scale must be %ld "
                         "(batch dim of x1 %ld + fixed dim %zu)",
                         expectedScaleDimNum, x1BatchDimNum, MX_X2_SCALE_DIM)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (static_cast<int64_t>(x1ScaleDimNum) != expectedScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x1Scale", FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x1Scale must be %ld "
                         "(batch dim of x1 %ld + fixed dim %zu)",
                         expectedScaleDimNum, x1BatchDimNum, MX_X1_SCALE_DIM)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInputDtypeValid(const QuantMatmulActivationQuantWeightNzParams& params,
                                               const char* apiName)
{
    if (!CheckType(params.x1->GetDataType(), X1_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "x1", op::ToString(params.x1->GetDataType()).GetString(),
                                              FormatString("the dtype of x1 must be in dtype support list %s",
                                                           op::ToString(X1_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.x2->GetDataType(), X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "x2", op::ToString(params.x2->GetDataType()).GetString(),
                                              FormatString("the dtype of x2 must be in dtype support list %s",
                                                           op::ToString(X2_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckMxfp8DtypeValid(const QuantMatmulActivationQuantWeightNzParams& params,
                                               const char* apiName)
{
    if (CheckInputDtypeValid(params, apiName) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x1Scale", op::ToString(params.x1Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x1Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x2Scale", op::ToString(params.x2Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x2Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("QuantMatmulActivationQuant CheckMxfp8DtypeValid success.");
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckMxfp4DtypeValid(const QuantMatmulActivationQuantWeightNzParams& params,
                                               const char* apiName)
{
    if (params.x1->GetDataType() != op::DataType::DT_FLOAT4_E2M1) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x1", op::ToString(params.x1->GetDataType()).GetString(),
            "when the quantization mode is mx and x2 is FP4, the dtype of x1 must be FLOAT4_E2M1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetDataType() != op::DataType::DT_FLOAT4_E2M1) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x2", op::ToString(params.x2->GetDataType()).GetString(),
            "when the quantization mode is mx and x1 is FP4, the dtype of x2 must be FLOAT4_E2M1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x1Scale", op::ToString(params.x1Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x1Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "x2Scale", op::ToString(params.x2Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x2Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("QuantMatmulActivationQuant CheckMxfp4DtypeValid success.");
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckDtype(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    auto x1ScaleDtype = params.x1Scale->GetDataType();
    auto x2ScaleDtype = params.x2Scale->GetDataType();
    auto yDtype = params.y->GetDataType();
    auto yScaleDtype = params.yScale->GetDataType();
    if (yDtype != static_cast<op::DataType>(params.y_dtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, "y", op::ToString(yDtype).GetString(),
            FormatString("the dtype of y must be %s, which is the same as x1",
                         op::ToString(static_cast<op::DataType>(params.y_dtype)).GetString())
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsMxFp8Input(params.x1, params.x2, params.y, params.yScale)) {
        CHECK_COND(IsMxQuantDim(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Check IsMxQuantDim failed.");
        if (params.bias != nullptr && params.bias->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "bias", op::ToString(params.bias->GetDataType()).GetString(),
                                                  "the dtype of bias must be FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return CheckMxfp8DtypeValid(params, apiName);
    } else if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        CHECK_COND(IsMxQuantDim(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Check IsMxQuantDim failed.");
        if (params.bias != nullptr && params.bias->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "bias", op::ToString(params.bias->GetDataType()).GetString(),
                                                  "the dtype of bias must be FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return CheckMxfp4DtypeValid(params, apiName);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName, "x1, x2, x1Scale, x2Scale, y, yScale",
            FormatString("%s, %s, %s, %s, %s, %s", op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                         op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString(),
                         op::ToString(yDtype).GetString(), op::ToString(yScaleDtype).GetString())
                .c_str(),
            FormatString(
                "when the dtypes of x1 and x2 are %s and %s, and the dtypes of x1Scale and x2Scale are %s "
                "and %s, and the dtypes of y and yScale are %s and %s, this dtype combination can not be supported",
                op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString(),
                op::ToString(yDtype).GetString(), op::ToString(yScaleDtype).GetString())
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
}

static inline aclnnStatus CheckOptionalAlg(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    CHECK_RET(params.activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const std::string activationType(params.activationType);
    if (activationType != "gelu_tanh" && activationType != "gelu_erf") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "activationType", activationType,
                                              "The activationType must be gelu_tanh or gelu_erf");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(params.quantMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const std::string quantMode(params.quantMode);
    if (quantMode != "mx") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "quantMode", quantMode, "The quantMode must be mx");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(params.roundMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    std::string roundMode(params.roundMode);
    if (roundMode != "rint" && roundMode != "floor" && roundMode != "round") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "roundMode", roundMode,
                                              "roundMode optional values are rint/floor/round, it's enabled when "
                                              "dynamic mx quant, fp8 only support rint, fp4 support all");
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool isMxFp4 = IsMxFp4Input(params.x1, params.x2, params.y, params.yScale);
    if (isMxFp4) {
        if (params.scaleAlg != 0 && params.scaleAlg != 2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "scaleAlg", std::to_string(params.scaleAlg),
                "when the dtypes of x1, x2 and y are FP4, the scaleAlg optional values are 0/2");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (params.scaleAlg != 0 && params.scaleAlg != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "scaleAlg", std::to_string(params.scaleAlg),
                "when the dtypes of x1 and x2 are not FP4, the scaleAlg optional values are 0/1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.scaleAlg == 2) {
        if (!IsFloatEqual(params.dstTypeMax, 0.0) && (params.dstTypeMax < 6.0 || params.dstTypeMax > 12.0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "dstTypeMax", std::to_string(params.dstTypeMax),
                                                  "when scaleAlg is 2, dstTypeMax must be 0.0 or in range [6.0, 12.0]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto yDtype = params.y->GetDataType();
    if (yDtype == op::DataType::DT_FLOAT4_E2M1) {
        if (roundMode != "rint" && roundMode != "floor" && roundMode != "round") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "roundMode", roundMode,
                "when the dtype of y is FLOAT4_E2M1, roundMode optional values are rint/floor/round");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (roundMode != "rint") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "roundMode", roundMode,
                                                  "when the dtype of y is not FLOAT4_E2M1, roundMode must be rint");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!CheckType(params.y->GetDataType(), Y_DTYPE_SUPPORT_LIST) && params.scaleAlg == 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "scaleAlg", std::to_string(params.scaleAlg),
                                              FormatString("scaleAlg can't be 1 when the dtype of y not in %s",
                                                           op::ToString(Y_DTYPE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckShapeInfoMatch(const QuantMatmulActivationQuantWeightNzParams& params,
                                              const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    int64_t x2KDim = params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - 1) :
                                          params.x2->GetViewShape().GetDim(
                                              x2DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM);
    if (shapeInfo.kDim != x2KDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName, "x1 K, x2 K",
                                               FormatString("%ld, %ld", shapeInfo.kDim, x2KDim).c_str(),
                                               "the K dimension of x1 and x2 must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline bool CheckMKN(int64_t m, int64_t k, int64_t n, const char* apiName)
{
    if (m <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x1 M", std::to_string(m).c_str(),
                                              "the M dimension of x1 must be positive");
        return false;
    }
    if (k <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "K", std::to_string(k).c_str(),
                                              "the K dimension of x1 and x2 must be positive");
        return false;
    }
    if (n <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x2 N", std::to_string(n).c_str(),
                                              "the N dimension of x2 must be positive");
        return false;
    }
    return true;
}

static inline aclnnStatus CheckMxScaleLastDim(const QuantMatmulActivationQuantWeightNzParams& params,
                                              const char* apiName)
{
    if (!IsMicroScaling(params.x1Scale, params.x2Scale)) {
        return ACLNN_SUCCESS;
    }

    auto scale1LastDimValue = params.x1Scale->GetViewShape().GetDim(params.x1Scale->GetViewShape().GetDimNum() - 1);
    auto scale2LastDimValue = params.x2Scale->GetViewShape().GetDim(params.x2Scale->GetViewShape().GetDimNum() - 1);
    if (scale1LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1Scale->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x1Scale must be %d",
                         MXFP_MULTI_BASE_SIZE)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale2LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x2Scale must be %d",
                         MXFP_MULTI_BASE_SIZE)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckExpectedShapes(const QuantMatmulActivationQuantWeightNzParams& params,
                                              const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    auto& x1View = params.x1->GetViewShape();
    auto& x2View = params.x2->GetViewShape();
    int64_t x1DimNum = x1View.GetDimNum();
    int64_t x2DimNum = x2View.GetDimNum();

    if (x1DimNum < MX_X1_DIM || x2DimNum < MX_X1_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "unsupported dim combination: x1DimNum=%ld, x2DimNum=%ld", x1DimNum, x2DimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t x1BatchCount = x1DimNum - 2;
    int64_t x2BatchCount = x2DimNum - 2;
    int64_t batchDimNum = std::max(x1BatchCount, x2BatchCount);
    for (int64_t i = 0; i < batchDimNum; ++i) {
        int64_t x1Idx = i - (batchDimNum - x1BatchCount);
        int64_t x2Idx = i - (batchDimNum - x2BatchCount);
        int64_t x1BatchDim = (x1Idx >= 0) ? x1View.GetDim(x1Idx) : 1;
        int64_t x2BatchDim = (x2Idx >= 0) ? x2View.GetDim(x2Idx) : 1;
        if (x1BatchDim != x2BatchDim && x1BatchDim != 1 && x2BatchDim != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch dim %ld mismatch: x1=%ld, x2=%ld", i, x1BatchDim, x2BatchDim);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    int64_t x1M = params.transposeX1 ? x1View.GetDim(x1DimNum - 1) : x1View.GetDim(x1DimNum - 2);
    int64_t x1K = params.transposeX1 ? x1View.GetDim(x1DimNum - 2) : x1View.GetDim(x1DimNum - 1);
    if (x1M != shapeInfo.mDim || x1K != shapeInfo.kDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName, "x1",
                                              StripEnclosingSquareBrackets(op::ToString(x1View).GetString()).c_str(),
                                              FormatString("x1 last two dims must be [%ld, %ld], but got [%ld, %ld]",
                                                           params.transposeX1 ? shapeInfo.kDim : shapeInfo.mDim,
                                                           params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim,
                                                           x1View.GetDim(x1DimNum - 2), x1View.GetDim(x1DimNum - 1))
                                                  .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t x2K = params.transposeX2 ? x2View.GetDim(x2DimNum - 1) : x2View.GetDim(x2DimNum - 2);
    int64_t x2N = params.transposeX2 ? x2View.GetDim(x2DimNum - 2) : x2View.GetDim(x2DimNum - 1);
    if (x2K != shapeInfo.kDim || x2N != shapeInfo.nDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName, "x2",
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
            apiName, "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1Scale->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x1Scale must be %s", op::ToString(x1ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetViewShape() != x2ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x2Scale must be %s", op::ToString(x2ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckOutputShape(const QuantMatmulActivationQuantWeightNzParams& params,
                                           const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    auto& yView = params.y->GetViewShape();
    int64_t yDimNum = yView.GetDimNum();
    if (yView.GetDim(yDimNum - 2) != shapeInfo.mDim || yView.GetDim(yDimNum - 1) != shapeInfo.nDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "y", StripEnclosingSquareBrackets(op::ToString(yView).GetString()).c_str(),
            FormatString("y last two dims must be [%ld, %ld], but got [%ld, %ld]", shapeInfo.mDim, shapeInfo.nDim,
                         yView.GetDim(yDimNum - 2), yView.GetDim(yDimNum - 1))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale) && shapeInfo.nDim % 2 != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "y N", std::to_string(shapeInfo.nDim).c_str(),
            "when the dtypes of x1, x2 and y are FP4, the N dimension of y must be even");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto& yScaleView = params.yScale->GetViewShape();
    int64_t yScaleDimNum = yScaleView.GetDimNum();
    int64_t expectedScaleN = Ops::Base::CeilDiv(shapeInfo.nDim, SPLIT_SIZE);
    if (yScaleView.GetDim(yScaleDimNum - 3) != shapeInfo.mDim ||
        yScaleView.GetDim(yScaleDimNum - 2) != expectedScaleN ||
        yScaleView.GetDim(yScaleDimNum - 1) != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "yScale", StripEnclosingSquareBrackets(op::ToString(yScaleView).GetString()).c_str(),
            FormatString("yScale last three dims must be [%ld, %ld, %ld], but got [%ld, %ld, %ld]", shapeInfo.mDim,
                         expectedScaleN, MXFP_MULTI_BASE_SIZE, yScaleView.GetDim(yScaleDimNum - 3),
                         yScaleView.GetDim(yScaleDimNum - 2), yScaleView.GetDim(yScaleDimNum - 1))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline bool CheckGroupSize(QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    auto groupSize = params.groupSize;
    if (groupSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "groupSize", std::to_string(groupSize).c_str(),
                                              "groupSize can not be negative");
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
            apiName, "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%ld, %lu, %lu, %lu", groupSize, groupSizeM, groupSizeN, groupSizeK).c_str(),
            "when the quantization mode is mx, groupSize must be 4295032864 and Torch API group_sizes must be [1, "
            "1, 32]");
        return false;
    }

    OP_LOGD("QuantMatmulActivationQuant check group_size success.");
    return true;
}
} // namespace QBMMActivationQuant
#endif
