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
#include "quant_matmul_activation_quant_checker.h"
#include "log/log.h"

namespace QBMMActivationQuant {
using namespace op;
using Ops::NN::FormatString;
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
constexpr uint32_t MX_X1_DIM_MAX = 6U;
constexpr uint32_t MX_X2_DIM = 2U;
constexpr uint32_t MX_X2_DIM_MIN = 4U;
constexpr uint32_t MX_X2_DIM_MAX = 8U;
constexpr uint32_t MX_X1_SCALE_DIM = 3U;
constexpr uint32_t MX_X2_SCALE_DIM = 3U;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr int64_t SWIGLU_BRANCH_COUNT = 2L;
constexpr int64_t SPLIT_SIZE = 64L;
static constexpr int PENULTIMATE_DIM = 2;

static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
// bits above groupSizeM|groupSizeN|groupSizeK must be zero
static const int32_t GROUP_RESERVED_BIT_OFFSET = 48;
static const int64_t PERGROUP_GROUP_SIZE = 32L;
static const size_t MX_SCALE_MAX_DIM = 3;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;
// two FLOAT4_E2M1 nibbles are packed into one byte
static constexpr int64_t FP4_PACK_RATIO = 2L;
// scaleAlg=2 (FP4 dynamic dtype range): 0 keeps the dtype default, otherwise [6, 12]
static constexpr float DST_TYPE_MAX_DISABLED = 0.0F;
static constexpr float DST_TYPE_MAX_MIN = 6.0F;
static constexpr float DST_TYPE_MAX_MAX = 12.0F;

static inline bool IsFloatEqual(float a, float b) { return std::abs(a - b) <= std::numeric_limits<float>::epsilon(); }

static inline bool IsSwiGluActivation(const QuantMatmulActivationQuantWeightNzParams& params)
{
    return params.activationType != nullptr && std::string(params.activationType) == "swiglu";
}

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
        OP_LOGD("Special case: transpose attribute does not need to be set.");
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
        OP_LOGD("Detected a transposed/non-contiguous tensor layout; swapping the last two dimensions.");
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
    OP_LOGD("Resolved transpose attributes: transposeX1=%s, transposeX2=%s", transposeX1 ? "true" : "false",
            transposeX2 ? "true" : "false");
}

static inline MatmulShapeInfo GetMatmulShapeInfo(const QuantMatmulActivationQuantWeightNzParams& params)
{
    int64_t x1DimNum = params.x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    return {
        params.transposeX1 ?
            params.x1->GetViewShape().GetDim(x1DimNum - LAST_FIRST_DIM_INDEX) :
            params.x1->GetViewShape().GetDim(x1DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM),
        params.transposeX1 ?
            params.x1->GetViewShape().GetDim(x1DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM) :
            params.x1->GetViewShape().GetDim(x1DimNum - LAST_FIRST_DIM_INDEX),
        params.transposeX2 ?
            params.x2->GetViewShape().GetDim(x2DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM) :
            params.x2->GetViewShape().GetDim(x2DimNum - LAST_FIRST_DIM_INDEX),
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

    x1ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < x1BatchDimNum; ++i) {
        x1ScaleExpectShape.AppendDim(x1View.GetDim(i));
    }
    if (params.transposeX1) {
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
    } else {
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
    }
    x1ScaleExpectShape.AppendDim(MXFP_MULTI_BASE_SIZE);

    x2ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < x2BatchDimNum; ++i) {
        x2ScaleExpectShape.AppendDim(x2View.GetDim(i));
    }
    if (params.transposeX2) {
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
    } else {
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE));
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
    }
    x2ScaleExpectShape.AppendDim(MXFP_MULTI_BASE_SIZE);
}

static inline int64_t InferOutputShape(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    int64_t inferredOutbatchValue = 1;
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
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "x1/x2 batch dim",
                FormatString("%lld, %lld", static_cast<long long>(shortDimValue), static_cast<long long>(longDimValue))
                    .c_str(),
                "the batch dimensions of x1 and x2 must be broadcastable");
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        if (shortDimValue <= 0 || longDimValue <= 0 ||
            inferredOutbatchValue > std::numeric_limits<uint32_t>::max() / curBatchValue) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "x1/x2 batch dim",
                FormatString("%lld, %lld", static_cast<long long>(shortDimValue), static_cast<long long>(longDimValue))
                    .c_str(),
                "the batch dimensions must be positive and the broadcast batch size must fit the uint32 tiling range");
            return OUTPUT_INFER_FAIL;
        }
        inferredOutbatchValue = inferredOutbatchValue * curBatchValue;
    }
    return inferredOutbatchValue;
}

static inline bool MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, aclOpExecutor* executor)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale tensor is absent or does not require contiguous conversion.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum -
                                                                 QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM);
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3); // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&            // 3： 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM] ==
            lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4：batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM, lastThirdDim);
        swapedShape.SetDim(dimNum - 3, lastSecondDim); // 3： 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    if (mxScaleTensor == nullptr) {
        return false;
    }
    return true;
}

static inline aclnnStatus IsMxQuantDim(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    int64_t x1DimNum = static_cast<int64_t>(params.x1->GetViewShape().GetDimNum());
    int64_t x2DimNum = static_cast<int64_t>(params.x2->GetViewShape().GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MX_X1_DIM), 0);
    int64_t x2BatchDimNum = std::max<int64_t>(x2DimNum - static_cast<int64_t>(MX_X2_DIM), 0);

    auto x1ScaleDimNum = params.x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    int64_t expectedX1ScaleDimNum = x1BatchDimNum + static_cast<int64_t>(MX_X1_SCALE_DIM);
    int64_t expectedX2ScaleDimNum = x2BatchDimNum + static_cast<int64_t>(MX_X2_SCALE_DIM);
    if (static_cast<int64_t>(x1ScaleDimNum) != expectedX1ScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x1Scale", FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the rank of x1Scale must be %lld "
                         "(x1 batch rank %lld + fixed rank %u)",
                         static_cast<long long>(expectedX1ScaleDimNum), static_cast<long long>(x1BatchDimNum),
                         MX_X1_SCALE_DIM)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (static_cast<int64_t>(x2ScaleDimNum) != expectedX2ScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x2Scale", FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the rank of x2Scale must be %lld "
                         "(x2 batch rank %lld + fixed rank %u)",
                         static_cast<long long>(expectedX2ScaleDimNum), static_cast<long long>(x2BatchDimNum),
                         MX_X2_SCALE_DIM)
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

static inline aclnnStatus CheckDtype(const QuantMatmulActivationQuantWeightNzParams& params, const char* apiName)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    auto x1ScaleDtype = params.x1Scale->GetDataType();
    auto x2ScaleDtype = params.x2Scale->GetDataType();
    auto yDtype = params.y->GetDataType();
    auto yScaleDtype = params.yScale->GetDataType();

    if (CheckInputDtypeValid(params, apiName) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (IsMxFp8Input(params.x1, params.x2, params.y, params.yScale) ||
        IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        CHECK_RET(IsMxQuantDim(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        if (params.bias != nullptr && params.bias->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "bias", op::ToString(params.bias->GetDataType()).GetString(),
                                                  "the dtype of bias must be FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1ScaleDtype != op::DataType::DT_FLOAT8_E8M0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                apiName, "x1Scale", op::ToString(x1ScaleDtype).GetString(),
                "when the quantization mode is mx, the dtype of x1Scale must be FLOAT8_E8M0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2ScaleDtype != op::DataType::DT_FLOAT8_E8M0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                apiName, "x2Scale", op::ToString(x2ScaleDtype).GetString(),
                "when the quantization mode is mx, the dtype of x2Scale must be FLOAT8_E8M0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (yDtype != static_cast<op::DataType>(params.y_dtype)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                apiName, "y", op::ToString(yDtype).GetString(),
                FormatString("the dtype of y must be %s, which is the same as x1",
                             op::ToString(static_cast<op::DataType>(params.y_dtype)).GetString())
                    .c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_LOGD("Dtype validation succeeded.");
        return ACLNN_SUCCESS;
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName, "x1, x2, x1Scale, x2Scale, y, yScale",
            FormatString("%s, %s, %s, %s, %s, %s", op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                         op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString(),
                         op::ToString(yDtype).GetString(), op::ToString(yScaleDtype).GetString())
                .c_str(),
            FormatString(
                "when the dtypes of x1 and x2 are %s and %s, and the dtypes of x1Scale and x2Scale are %s "
                "and %s, and the dtypes of y and yScale are %s and %s; this dtype combination is not supported",
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
    if (activationType != "gelu_tanh" && activationType != "gelu_erf" && activationType != "swiglu") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "activationType", activationType,
                                              "The activationType must be gelu_tanh, gelu_erf or swiglu");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isWeightNz = params.x2 != nullptr &&
                            ge::GetPrimaryFormat(params.x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    if (activationType == "swiglu" &&
        ((params.transposeX1 && isWeightNz) || (params.scaleAlg != 0 && params.scaleAlg != 1) ||
         (params.roundMode != nullptr && std::string(params.roundMode) != "rint"))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "swiglu constraints", "unsupported",
                                              "SwiGLU MX requires transposeX1=false for WeightNZ, scaleAlg=0/1 and "
                                              "roundMode=rint");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (activationType == "swiglu" && (!IsMxFp8Input(params.x1, params.x2, params.y, params.yScale))) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "x1/x2/y", "non-MXFP8", "SwiGLU MX only supports MXFP8");
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
                                              "roundMode must be one of rint, floor, or round. FP8 supports only "
                                              "rint; FP4 supports all three modes.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool isMxFp4 = IsMxFp4Input(params.x1, params.x2, params.y, params.yScale);
    if (isMxFp4) {
        if (params.scaleAlg != 0 && params.scaleAlg != 2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "scaleAlg", std::to_string(params.scaleAlg),
                                                  "when the dtypes of x1, x2 and y are FP4, scaleAlg must be 0 or 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (params.scaleAlg != 0 && params.scaleAlg != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "scaleAlg", std::to_string(params.scaleAlg),
                                                  "when the dtypes of x1 and x2 are not FP4, scaleAlg must be 0 or 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.scaleAlg == 2) {
        if (!IsFloatEqual(params.dstTypeMax, DST_TYPE_MAX_DISABLED) &&
            !(params.dstTypeMax >= DST_TYPE_MAX_MIN && params.dstTypeMax <= DST_TYPE_MAX_MAX)) {
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
                "when the dtype of y is FLOAT4_E2M1, roundMode must be rint, floor, or round");
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
                                              FormatString("scaleAlg cannot be 1 when the dtype of y is not in %s",
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
    int64_t x2KDim = params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - LAST_FIRST_DIM_INDEX) :
                                          params.x2->GetViewShape().GetDim(
                                              x2DimNum - QuantMatmulActivationQuantAclnnCheck::PENULTIMATE_DIM);
    if (shapeInfo.kDim != x2KDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, "x1 K, x2 K",
            FormatString("%lld, %lld", static_cast<long long>(shapeInfo.kDim), static_cast<long long>(x2KDim)).c_str(),
            "the K dimension of x1 and x2 must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline bool CheckMKN(int64_t m, int64_t k, int64_t n, const char* apiName)
{
    if (m > std::numeric_limits<uint32_t>::max() || k > std::numeric_limits<uint32_t>::max() ||
        n > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName, "M, K, N",
                                               FormatString("%lld, %lld, %lld", static_cast<long long>(m),
                                                            static_cast<long long>(k), static_cast<long long>(n))
                                                   .c_str(),
                                               "M, K and N must fit the uint32 tiling fields");
        return false;
    }
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
            apiName, "x1Scale", op::ToString(params.x1Scale->GetViewShape()).GetString(),
            FormatString("when the quantization mode is mx, the last dimension of x1Scale must be %lld",
                         static_cast<long long>(MXFP_MULTI_BASE_SIZE))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale2LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2Scale", op::ToString(params.x2Scale->GetViewShape()).GetString(),
            FormatString("when the quantization mode is mx, the last dimension of x2Scale must be %lld",
                         static_cast<long long>(MXFP_MULTI_BASE_SIZE))
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

    if (x1DimNum < static_cast<int64_t>(MX_X1_DIM) || x2DimNum < static_cast<int64_t>(MX_X2_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            apiName, "x1, x2",
            FormatString("%lldD, %lldD", static_cast<long long>(x1DimNum), static_cast<long long>(x2DimNum)).c_str(),
            "the ranks of x1 and x2 must be at least 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (InferOutputShape(params, apiName) == OUTPUT_INFER_FAIL) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t x1BatchCount = x1DimNum - static_cast<int64_t>(MX_X1_DIM);
    int64_t x2BatchCount = x2DimNum - static_cast<int64_t>(MX_X2_DIM);
    int64_t batchDimNum = std::max(x1BatchCount, x2BatchCount);
    for (int64_t i = 0; i < batchDimNum; ++i) {
        int64_t x1Idx = i - (batchDimNum - x1BatchCount);
        int64_t x2Idx = i - (batchDimNum - x2BatchCount);
        int64_t x1BatchDim = (x1Idx >= 0) ? x1View.GetDim(x1Idx) : 1;
        int64_t x2BatchDim = (x2Idx >= 0) ? x2View.GetDim(x2Idx) : 1;
        if (x1BatchDim != x2BatchDim && x1BatchDim != 1 && x2BatchDim != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "x1/x2 batch dim",
                FormatString("dim %lld: %lld, %lld", static_cast<long long>(i), static_cast<long long>(x1BatchDim),
                             static_cast<long long>(x2BatchDim))
                    .c_str(),
                "the batch dimensions of x1 and x2 must be broadcastable");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    int64_t x1M = params.transposeX1 ? x1View.GetDim(x1DimNum - LAST_FIRST_DIM_INDEX) :
                                       x1View.GetDim(x1DimNum - LAST_SECOND_DIM_INDEX);
    int64_t x1K = params.transposeX1 ? x1View.GetDim(x1DimNum - LAST_SECOND_DIM_INDEX) :
                                       x1View.GetDim(x1DimNum - LAST_FIRST_DIM_INDEX);
    if (x1M != shapeInfo.mDim || x1K != shapeInfo.kDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x1", op::ToString(x1View).GetString(),
            FormatString("x1 last two dims must be [%lld, %lld], but got [%lld, %lld]",
                         static_cast<long long>(params.transposeX1 ? shapeInfo.kDim : shapeInfo.mDim),
                         static_cast<long long>(params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim),
                         static_cast<long long>(x1View.GetDim(x1DimNum - LAST_SECOND_DIM_INDEX)),
                         static_cast<long long>(x1View.GetDim(x1DimNum - LAST_FIRST_DIM_INDEX)))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t x2K = params.transposeX2 ? x2View.GetDim(x2DimNum - LAST_FIRST_DIM_INDEX) :
                                       x2View.GetDim(x2DimNum - LAST_SECOND_DIM_INDEX);
    int64_t x2N = params.transposeX2 ? x2View.GetDim(x2DimNum - LAST_SECOND_DIM_INDEX) :
                                       x2View.GetDim(x2DimNum - LAST_FIRST_DIM_INDEX);
    if (x2K != shapeInfo.kDim || x2N != shapeInfo.nDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2", op::ToString(x2View).GetString(),
            FormatString("x2 last two dims must be [%lld, %lld], but got [%lld, %lld]",
                         static_cast<long long>(params.transposeX2 ? shapeInfo.nDim : shapeInfo.kDim),
                         static_cast<long long>(params.transposeX2 ? shapeInfo.kDim : shapeInfo.nDim),
                         static_cast<long long>(x2View.GetDim(x2DimNum - LAST_SECOND_DIM_INDEX)),
                         static_cast<long long>(x2View.GetDim(x2DimNum - LAST_FIRST_DIM_INDEX)))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShape(params, shapeInfo, x1ScaleExpectShape, x2ScaleExpectShape);

    if (params.x1Scale->GetViewShape() != x1ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x1Scale", op::ToString(params.x1Scale->GetViewShape()).GetString(),
            FormatString("the shape of x1Scale must be %s", op::ToString(x1ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetViewShape() != x2ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2Scale", op::ToString(params.x2Scale->GetViewShape()).GetString(),
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
    const auto& x1View = params.x1->GetViewShape();
    const auto& x2View = params.x2->GetViewShape();
    const int64_t expectedRank = std::max(x1View.GetDimNum(), x2View.GetDimNum());
    const auto& yScaleView = params.yScale->GetViewShape();
    if (yDimNum != expectedRank || yScaleView.GetDimNum() != static_cast<size_t>(expectedRank + 1)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, "y/yScale rank",
            FormatString("%lld, %lld", static_cast<long long>(yDimNum), static_cast<long long>(yScaleView.GetDimNum()))
                .c_str(),
            FormatString("the ranks of y and yScale must be %lld and %lld", static_cast<long long>(expectedRank),
                         static_cast<long long>(expectedRank + 1))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (int64_t i = 0; i < expectedRank - 2; ++i) {
        const int64_t x1Index = i - (expectedRank - static_cast<int64_t>(x1View.GetDimNum()));
        const int64_t x2Index = i - (expectedRank - static_cast<int64_t>(x2View.GetDimNum()));
        const int64_t batchDim = std::max(x1Index >= 0 ? x1View.GetDim(x1Index) : 1,
                                          x2Index >= 0 ? x2View.GetDim(x2Index) : 1);
        if (yView.GetDim(i) != batchDim || yScaleView.GetDim(i) != batchDim) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "y/yScale batch dim",
                FormatString("dim %lld: %lld, %lld, expected %lld", static_cast<long long>(i),
                             static_cast<long long>(yView.GetDim(i)), static_cast<long long>(yScaleView.GetDim(i)),
                             static_cast<long long>(batchDim))
                    .c_str(),
                "the batch dimensions of y and yScale must match the broadcast batch dimensions of x1 and x2");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t outputN = IsSwiGluActivation(params) ? shapeInfo.nDim / SWIGLU_BRANCH_COUNT : shapeInfo.nDim;
    if (IsSwiGluActivation(params) && (shapeInfo.nDim <= 0 || shapeInfo.nDim % SPLIT_SIZE != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x2 N", std::to_string(shapeInfo.nDim).c_str(),
                                              "SwiGLU requires positive pre-activation N divisible by 64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yView.GetDim(yDimNum - LAST_SECOND_DIM_INDEX) != shapeInfo.mDim ||
        yView.GetDim(yDimNum - LAST_FIRST_DIM_INDEX) != outputN) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "y", op::ToString(yView).GetString(),
            FormatString("y last two dims must be [%lld, %lld], but got [%lld, %lld]",
                         static_cast<long long>(shapeInfo.mDim), static_cast<long long>(outputN),
                         static_cast<long long>(yView.GetDim(yDimNum - LAST_SECOND_DIM_INDEX)),
                         static_cast<long long>(yView.GetDim(yDimNum - LAST_FIRST_DIM_INDEX)))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale) && shapeInfo.nDim % FP4_PACK_RATIO != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "y N", std::to_string(shapeInfo.nDim).c_str(),
            "when the dtypes of x1, x2 and y are FP4, the N dimension of y must be even");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t yScaleDimNum = yScaleView.GetDimNum();
    int64_t expectedScaleN = Ops::Base::CeilDiv(outputN, SPLIT_SIZE);
    if (yScaleView.GetDim(yScaleDimNum - LAST_THIRD_DIM_INDEX) != shapeInfo.mDim ||
        yScaleView.GetDim(yScaleDimNum - LAST_SECOND_DIM_INDEX) != expectedScaleN ||
        yScaleView.GetDim(yScaleDimNum - LAST_FIRST_DIM_INDEX) != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "yScale", op::ToString(yScaleView).GetString(),
            FormatString("yScale last three dims must be [%lld, %lld, %lld], but got [%lld, %lld, %lld]",
                         static_cast<long long>(shapeInfo.mDim), static_cast<long long>(expectedScaleN),
                         static_cast<long long>(MXFP_MULTI_BASE_SIZE),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_THIRD_DIM_INDEX)),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_SECOND_DIM_INDEX)),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_FIRST_DIM_INDEX)))
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
                                              "groupSize cannot be negative");
        return false;
    }
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;

    if (groupSize == 0) {
        params.groupSize = (1UL << GROUP_M_OFFSET) | (1UL << GROUP_N_OFFSET) |
                           static_cast<uint64_t>(PERGROUP_GROUP_SIZE);
    } else if ((static_cast<uint64_t>(groupSize) >> GROUP_RESERVED_BIT_OFFSET) != 0 ||
               groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1UL || groupSizeN != 1UL) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%lld, %llu, %llu, %llu", static_cast<long long>(groupSize),
                         static_cast<unsigned long long>(groupSizeM), static_cast<unsigned long long>(groupSizeN),
                         static_cast<unsigned long long>(groupSizeK))
                .c_str(),
            "when the quantization mode is mx, groupSize must be 4295032864 and Torch API group_sizes must be [1, "
            "1, 32]");
        return false;
    }

    OP_LOGD("Group-size validation succeeded.");
    return true;
}
} // namespace QBMMActivationQuant
#endif
