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

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "base/err_msg.h"
#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/quant_matmul_util.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_matmul_activation_quant.h"
#include "quant_matmul_activation_quant_common.h"
#include "quant_matmul_activation_quant_checker.h"

namespace QBMMActivationQuant {
using namespace op;
using Ops::NN::FormatString;

struct MatmulShapeInfo {
    int64_t mDim;
    int64_t kDim;
    int64_t nDim;
};

static const std::initializer_list<op::DataType> X1_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> X2_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};

constexpr uint32_t MATRIX_DIM_COUNT = 2U;
constexpr uint32_t MIN_MATRIX_RANK = 2U;
constexpr uint32_t MAX_MATRIX_RANK = 6U;
constexpr uint32_t MIN_NZ_STORAGE_RANK = 4U;
constexpr uint32_t MAX_NZ_STORAGE_RANK = 8U;
constexpr uint32_t MX_SCALE_DIM_COUNT = 3U;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MX_SCALE_STORAGE_GROUP = 64L;
constexpr int64_t MX_SCALE_PAIR_SIZE = 2L;
constexpr int64_t SWIGLU_BRANCH_COUNT = 2L;

constexpr int32_t GROUP_M_OFFSET = 32;
constexpr int32_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
// bits above groupSizeM|groupSizeN|groupSizeK must be zero
constexpr int32_t GROUP_RESERVED_BIT_OFFSET = 48;
constexpr int64_t PERGROUP_GROUP_SIZE = 32L;
constexpr size_t MX_SCALE_MIN_RANK = 3;
constexpr int64_t BATCH_INFER_FAILED = -1L;
// two FLOAT4_E2M1 nibbles are packed into one byte
constexpr int64_t FP4_PACK_RATIO = 2L;
// scaleAlg=2 (FP4 dynamic dtype range): 0 keeps the dtype default, otherwise [6, 12]
constexpr float DST_TYPE_MAX_DISABLED = 0.0F;
constexpr float DST_TYPE_MAX_MIN = 6.0F;
constexpr float DST_TYPE_MAX_MAX = 12.0F;

static inline bool IsSwiGluActivation(const QuantMatmulActivationQuantParams& params)
{
    return params.activationType != nullptr && std::string(params.activationType) == "swiglu";
}

static inline aclnnStatus CheckPlatformAndInputs(const aclTensor* x1, const aclTensor* x2, const char* apiName)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        SocVersion socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_RUNTIME_ERROR, "SOC version %s is not supported",
                               op::ToString(socVersion).GetString());
        const std::string msg = std::string("SOC version ") + op::ToString(socVersion).GetString() +
                                " is not supported";
        const std::vector<const char*> msgKey = {"message"};
        const std::vector<const char*> msgValue = {msg.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1001", msgKey, msgValue);
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    if (x1 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x1", "null", "x1 cannot be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x2", "null", "x2 cannot be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckNotNull(const QuantMatmulActivationQuantParams& params)
{
    OP_CHECK_NULL(params.x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x1Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.y, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.yScale, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool IsNdCompatibleBaseFormat(op::Format format)
{
    const std::initializer_list<op::Format> supportedFormats = {
        Format::FORMAT_ND,    Format::FORMAT_NCHW,  Format::FORMAT_NHWC, Format::FORMAT_HWCN,
        Format::FORMAT_NCDHW, Format::FORMAT_NDHWC, Format::FORMAT_NCL};
    return std::find(supportedFormats.begin(), supportedFormats.end(), format) != supportedFormats.end();
}

static inline aclnnStatus CheckOutputLayout(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    const std::pair<const aclTensor*, const char*> outputs[] = {{params.y, "y"}, {params.yScale, "yScale"}};
    for (const auto& output : outputs) {
        const auto format = output.first->GetStorageFormat();
        if (!IsNdCompatibleBaseFormat(format)) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                apiName, output.second, op::ToString(format).GetString(),
                "the output format must be ND or an ND-compatible base format; private formats are not supported");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!op::IsContiguous(output.first)) {
            OP_LOGE_FOR_INVALID_STRIDE(apiName, output.second, op::ToString(output.first->GetViewStrides()).GetString(),
                                       "contiguous");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInputFormats(const QuantMatmulActivationQuantParams& params, op::Format x2Format,
                                            const char* apiName)
{
    if (params.x1->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(apiName, "x1", op::ToString(params.x1->GetStorageFormat()).GetString(),
                                                "the format of x1 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetStorageFormat() != x2Format) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            apiName, "x2", op::ToString(params.x2->GetStorageFormat()).GetString(),
            x2Format == Format::FORMAT_ND ? "the format of x2 must be ND" : "the format of x2 must be FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(apiName, "x1Scale",
                                                op::ToString(params.x1Scale->GetStorageFormat()).GetString(),
                                                "the format of x1Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(apiName, "x2Scale",
                                                op::ToString(params.x2Scale->GetStorageFormat()).GetString(),
                                                "the format of x2Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.bias != nullptr && params.bias->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(apiName, "bias",
                                                op::ToString(params.bias->GetStorageFormat()).GetString(),
                                                "the format of bias must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline bool GetPositiveShapeSize(const op::Shape& shape, int64_t& shapeSize)
{
    shapeSize = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const auto dim = shape.GetDim(i);
        if (dim <= 0 || shapeSize > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        shapeSize *= dim;
    }
    return true;
}

static inline bool CheckViewRange(const aclTensor* tensor, int64_t storageSize)
{
    const auto& shape = tensor->GetViewShape();
    const auto& strides = tensor->GetViewStrides();
    const auto offset = tensor->GetViewOffset();
    if (strides.size() != shape.GetDimNum() || offset < 0 || offset >= storageSize) {
        return false;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        // NZ consumes the full padded physical shape, not an ND strided slice.
        const auto& storageShape = tensor->GetStorageShape();
        if (offset != 0 || shape.GetDimNum() < LAST_SECOND_DIM_INDEX ||
            storageShape.GetDimNum() != shape.GetDimNum() + LAST_SECOND_DIM_INDEX) {
            return false;
        }
        int64_t logicalBatchStride = 0;
        if (!GetPositiveShapeSize(shape, logicalBatchStride)) {
            return false;
        }
        int64_t physicalBatchStride = storageSize;
        bool matchesLogicalStrides = true;
        bool matchesPhysicalStrides = true;
        for (size_t i = 0; i + LAST_SECOND_DIM_INDEX < shape.GetDimNum(); ++i) {
            const auto batchDim = shape.GetDim(i);
            if (batchDim != storageShape.GetDim(i) || strides[i] < 0) {
                return false;
            }
            logicalBatchStride /= batchDim;
            physicalBatchStride /= batchDim;
            // A singleton batch axis does not advance the data address.
            if (batchDim > 1) {
                matchesLogicalStrides = matchesLogicalStrides && strides[i] == logicalBatchStride;
                matchesPhysicalStrides = matchesPhysicalStrides && strides[i] == physicalBatchStride;
            }
        }
        // PyTorch preserves logical strides during format cast. Also accept a complete physical-stride description.
        return matchesLogicalStrides || matchesPhysicalStrides;
    }
    if (tensor->GetDataType() == DataType::DT_FLOAT4_E2M1 && offset % FP4_PACK_RATIO != 0) {
        return false; // The data pointer must start at a whole byte.
    }
    int64_t remaining = storageSize - offset - 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0 || strides[i] < 0) {
            return false;
        }
        const auto extent = shape.GetDim(i) - 1;
        // Check against the remaining address range before multiplying or adding.
        if (extent != 0 && strides[i] > remaining / extent) {
            return false;
        }
        remaining -= extent * strides[i];
    }
    return true;
}

static inline aclnnStatus CheckTensorViews(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    // 空tensor 处理
    if (params.x1->IsEmpty() || params.x2->IsEmpty() || (params.x1Scale != nullptr && params.x1Scale->IsEmpty()) ||
        (params.x2Scale != nullptr && params.x2Scale->IsEmpty()) ||
        (params.bias != nullptr && params.bias->IsEmpty()) || (params.x3 != nullptr && params.x3->IsEmpty()) ||
        params.y->IsEmpty() || params.yScale->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            apiName, "x1, x2, x1Scale, x2Scale, bias, x3, y, yScale",
            Ops::NN::FormatString(
                "%s, %s, %s, %s, %s, %s, %s, %s", op::ToString(params.x1->GetViewShape()).GetString(),
                op::ToString(params.x2->GetViewShape()).GetString(),
                params.x1Scale != nullptr ? op::ToString(params.x1Scale->GetViewShape()).GetString() : "null",
                op::ToString(params.x2Scale->GetViewShape()).GetString(),
                params.bias != nullptr ? op::ToString(params.bias->GetViewShape()).GetString() : "null",
                params.x3 != nullptr ? op::ToString(params.x3->GetViewShape()).GetString() : "null",
                op::ToString(params.y->GetViewShape()).GetString(),
                op::ToString(params.yScale->GetViewShape()).GetString())
                .c_str(),
            Ops::NN::FormatString("The shapes of %s cannot be %s", "x1, x2, x1Scale, x2Scale, bias, x3, y, yScale",
                                  "empty")
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Validate metadata before CreateView and contiguous-layout detection perform shape arithmetic.
    for (const auto* input : {params.x1, params.x2, params.x1Scale, params.x2Scale, params.bias, params.x3,
                              static_cast<const aclTensor*>(params.y), static_cast<const aclTensor*>(params.yScale)}) {
        if (input == nullptr) {
            continue;
        }
        const auto& shape = input->GetViewShape();
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            if (shape.GetDim(i) <= 0 || shape.GetDim(i) > std::numeric_limits<uint32_t>::max()) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    apiName, "tensor", op::ToString(shape).GetString(),
                    "tensor dimensions must be positive and fit the uint32 tiling range");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        int64_t viewSize = 0;
        int64_t storageSize = 0;
        if (!GetPositiveShapeSize(shape, viewSize) || !GetPositiveShapeSize(input->GetStorageShape(), storageSize)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName, "tensor view/storage",
                FormatString("%s, %s", op::ToString(shape).GetString(),
                             op::ToString(input->GetStorageShape()).GetString())
                    .c_str(),
                "view and storage dimensions must be positive and their element counts must fit int64");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!CheckViewRange(input, storageSize)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "tensor view metadata", "invalid",
                "ND views must have matching stride rank, nonnegative strides and all addressed elements within "
                "storage; FP4 offsets must be even; FRACTAL_NZ views require offset 0 and batch strides consistently "
                "matching either the logical shape or the padded physical storage");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return CheckOutputLayout(params, apiName);
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

static inline MatmulShapeInfo GetMatmulShapeInfo(const QuantMatmulActivationQuantParams& params)
{
    int64_t x1DimNum = params.x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    return {
        params.transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - LAST_FIRST_DIM_INDEX) :
                             params.x1->GetViewShape().GetDim(x1DimNum - LAST_SECOND_DIM_INDEX),
        params.transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - LAST_SECOND_DIM_INDEX) :
                             params.x1->GetViewShape().GetDim(x1DimNum - LAST_FIRST_DIM_INDEX),
        params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - LAST_SECOND_DIM_INDEX) :
                             params.x2->GetViewShape().GetDim(x2DimNum - LAST_FIRST_DIM_INDEX),
    };
}

static inline void GetExpectedScaleShapes(const QuantMatmulActivationQuantParams& params,
                                          const MatmulShapeInfo& shapeInfo, op::Shape& x1ScaleExpectShape,
                                          op::Shape& x2ScaleExpectShape)
{
    // CheckDtype has already validated the mandatory E8M0 input scales.
    const auto& x1View = params.x1->GetViewShape();
    const auto& x2View = params.x2->GetViewShape();
    int64_t x1DimNum = static_cast<int64_t>(x1View.GetDimNum());
    int64_t x2DimNum = static_cast<int64_t>(x2View.GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MATRIX_DIM_COUNT), 0);
    int64_t x2BatchDimNum = std::max<int64_t>(x2DimNum - static_cast<int64_t>(MATRIX_DIM_COUNT), 0);

    x1ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < x1BatchDimNum; ++i) {
        x1ScaleExpectShape.AppendDim(x1View.GetDim(i));
    }
    if (params.transposeX1) {
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, MX_SCALE_STORAGE_GROUP));
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
    } else {
        x1ScaleExpectShape.AppendDim(shapeInfo.mDim);
        x1ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, MX_SCALE_STORAGE_GROUP));
    }
    x1ScaleExpectShape.AppendDim(MX_SCALE_PAIR_SIZE);

    x2ScaleExpectShape = op::Shape();
    for (int64_t i = 0; i < x2BatchDimNum; ++i) {
        x2ScaleExpectShape.AppendDim(x2View.GetDim(i));
    }
    if (params.transposeX2) {
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, MX_SCALE_STORAGE_GROUP));
    } else {
        x2ScaleExpectShape.AppendDim(Ops::Base::CeilDiv(shapeInfo.kDim, MX_SCALE_STORAGE_GROUP));
        x2ScaleExpectShape.AppendDim(shapeInfo.nDim);
    }
    x2ScaleExpectShape.AppendDim(MX_SCALE_PAIR_SIZE);
}

static inline int64_t InferOutputBatchSize(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    int64_t outputBatchSize = 1;
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    auto& higherRankTensor = x1DimNum > x2DimNum ? params.x1 : params.x2;
    auto& lowerRankTensor = x1DimNum > x2DimNum ? params.x2 : params.x1;
    size_t batchRankOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i + LAST_SECOND_DIM_INDEX < outDimNum; i++) {
        auto lowerRankDim = i < batchRankOffset ? 1 : lowerRankTensor->GetViewShape().GetDim(i - batchRankOffset);
        auto higherRankDim = higherRankTensor->GetViewShape().GetDim(i);
        if (lowerRankDim > 1 && higherRankDim > 1 && lowerRankDim != higherRankDim) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "x1/x2 batch dim",
                FormatString("%lld, %lld", static_cast<long long>(lowerRankDim), static_cast<long long>(higherRankDim))
                    .c_str(),
                "the batch dimensions of x1 and x2 must be broadcastable");
            return BATCH_INFER_FAILED;
        }
        int64_t batchDim = static_cast<int64_t>(std::max(lowerRankDim, higherRankDim));
        if (lowerRankDim <= 0 || higherRankDim <= 0 ||
            outputBatchSize > std::numeric_limits<uint32_t>::max() / batchDim) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, "x1/x2 batch dim",
                FormatString("%lld, %lld", static_cast<long long>(lowerRankDim), static_cast<long long>(higherRankDim))
                    .c_str(),
                "the batch dimensions must be positive and the broadcast batch size must fit the uint32 tiling range");
            return BATCH_INFER_FAILED;
        }
        outputBatchSize = outputBatchSize * batchDim;
    }
    return outputBatchSize;
}

static inline aclnnStatus CheckBiasShape(const QuantMatmulActivationQuantParams& params, int64_t nDim,
                                         const char* apiName)
{
    if (params.bias == nullptr) {
        return ACLNN_SUCCESS;
    }

    auto biasDimNum = params.bias->GetViewShape().GetDimNum();
    auto outDimNum = params.y->GetViewShape().GetDimNum();
    if (biasDimNum != 1 && biasDimNum != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName, "bias", FormatString("%zuD", biasDimNum).c_str(),
                                                 "the rank of bias must be 1 or 3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (biasDimNum == 1) {
        if (params.bias->GetViewShape().GetDim(0) != nDim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                apiName, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                FormatString("the shape of bias must be [%lld]", static_cast<long long>(nDim)).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (outDimNum == 2 || outDimNum == 4 || outDimNum == 5 || outDimNum == 6) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                apiName, "bias", FormatString("%zuD", biasDimNum).c_str(),
                FormatString("when output rank is %zu, only 1D bias is supported, but got 3D", outDimNum).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.bias->GetViewShape().GetDim(1) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName, "bias",
                                                  op::ToString(params.bias->GetViewShape()).GetString(),
                                                  "the 2nd dimension of bias must be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.bias->GetViewShape().GetDim(2) != nDim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                apiName, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                FormatString("the 3rd dimension of bias must be %lld", static_cast<long long>(nDim)).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t outputBatchSize = InferOutputBatchSize(params, apiName);
        if (outputBatchSize == BATCH_INFER_FAILED) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.bias->GetViewShape().GetDim(0) != outputBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                apiName, "bias", op::ToString(params.bias->GetViewShape()).GetString(),
                FormatString("the 1st dimension of bias must be %lld", static_cast<long long>(outputBatchSize))
                    .c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckMxScaleRanks(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    int64_t x1DimNum = static_cast<int64_t>(params.x1->GetViewShape().GetDimNum());
    int64_t x2DimNum = static_cast<int64_t>(params.x2->GetViewShape().GetDimNum());
    int64_t x1BatchDimNum = std::max<int64_t>(x1DimNum - static_cast<int64_t>(MATRIX_DIM_COUNT), 0);
    int64_t x2BatchDimNum = std::max<int64_t>(x2DimNum - static_cast<int64_t>(MATRIX_DIM_COUNT), 0);

    auto x1ScaleDimNum = params.x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    int64_t expectedX1ScaleDimNum = x1BatchDimNum + static_cast<int64_t>(MX_SCALE_DIM_COUNT);
    int64_t expectedX2ScaleDimNum = x2BatchDimNum + static_cast<int64_t>(MX_SCALE_DIM_COUNT);
    if (static_cast<int64_t>(x1ScaleDimNum) != expectedX1ScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x1Scale", FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the rank of x1Scale must be %lld "
                         "(x1 batch rank %lld + fixed rank %u)",
                         static_cast<long long>(expectedX1ScaleDimNum), static_cast<long long>(x1BatchDimNum),
                         MX_SCALE_DIM_COUNT)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (static_cast<int64_t>(x2ScaleDimNum) != expectedX2ScaleDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, "x2Scale", FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the rank of x2Scale must be %lld "
                         "(x2 batch rank %lld + fixed rank %u)",
                         static_cast<long long>(expectedX2ScaleDimNum), static_cast<long long>(x2BatchDimNum),
                         MX_SCALE_DIM_COUNT)
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInputDtypeValid(const QuantMatmulActivationQuantParams& params, const char* apiName)
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

static inline aclnnStatus CheckDtype(const QuantMatmulActivationQuantParams& params, const char* apiName)
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
        CHECK_RET(CheckMxScaleRanks(params, apiName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
        if (yDtype != static_cast<op::DataType>(params.yDtype)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                apiName, "y", op::ToString(yDtype).GetString(),
                FormatString("the dtype of y must be %s, which is the same as x1",
                             op::ToString(static_cast<op::DataType>(params.yDtype)).GetString())
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

// A supplied x3 is the multiplier for the sigmoid epilogue.
static inline aclnnStatus CheckX3(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    const bool sigmoid = params.activationType != nullptr && std::string(params.activationType) == "sigmoid";
    if (sigmoid != (params.x3 != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x3", "invalid presence",
                                              "x3 is required for sigmoid multiplication and must be absent "
                                              "for other activations");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x3 != nullptr && (params.x1->GetViewShape().GetDimNum() != MATRIX_DIM_COUNT ||
                                 params.x2->GetViewShape().GetDimNum() != MATRIX_DIM_COUNT ||
                                 params.x3->GetViewShape().GetDimNum() != MATRIX_DIM_COUNT ||
                                 params.y->GetViewShape().GetDimNum() != MATRIX_DIM_COUNT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "x1/x2/x3/y ranks", "batch is not supported",
                                              "x1, x2, x3 and y must have 2D logical shapes when x3 is provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x3 != nullptr &&
        (params.x3->GetDataType() != DataType::DT_BF16 || !IsNdCompatibleBaseFormat(params.x3->GetStorageFormat()) ||
         params.x3->GetViewShape() != params.y->GetViewShape())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "x3", "invalid tensor",
            "x3 must be BF16 with an ND-compatible base format and the same shape as y");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckAttributes(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    CHECK_RET(params.activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const std::string activationType(params.activationType);
    if (activationType != "gelu_tanh" && activationType != "gelu_erf" && activationType != "swiglu" &&
        activationType != "silu" && activationType != "sigmoid") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "activationType", activationType,
            "The activationType must be gelu_tanh, gelu_erf, swiglu, silu or sigmoid");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isWeightNz = params.x2 != nullptr &&
                            ge::GetPrimaryFormat(params.x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    const bool isFp8OnlyActivation = activationType == "swiglu" || activationType == "silu" ||
                                     activationType == "sigmoid";
    if (isFp8OnlyActivation && ((params.transposeX1 && isWeightNz) || (params.scaleAlg != 0 && params.scaleAlg != 1) ||
                                (params.roundMode != nullptr && std::string(params.roundMode) != "rint"))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "activation constraints", "unsupported",
                                              "SwiGLU, SiLU and sigmoid MX require transposeX1=false for WeightNZ, "
                                              "scaleAlg=0/1 and roundMode=rint");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isFp8OnlyActivation && (!IsMxFp8Input(params.x1, params.x2, params.y, params.yScale))) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, "x1/x2/y", "non-MXFP8",
                                              "SwiGLU, SiLU and sigmoid MX only support MXFP8");
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
        if (params.dstTypeMax != DST_TYPE_MAX_DISABLED &&
            !(params.dstTypeMax >= DST_TYPE_MAX_MIN && params.dstTypeMax <= DST_TYPE_MAX_MAX)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "dstTypeMax", std::to_string(params.dstTypeMax),
                                                  "when scaleAlg is 2, dstTypeMax must be 0.0 or in range [6.0, 12.0]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!isMxFp4 && roundMode != "rint") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "roundMode", roundMode,
                                              "when the dtype of y is not FLOAT4_E2M1, roundMode must be rint");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckKDimensionsMatch(const QuantMatmulActivationQuantParams& params,
                                                const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    int64_t x2KDim = params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - LAST_FIRST_DIM_INDEX) :
                                          params.x2->GetViewShape().GetDim(x2DimNum - LAST_SECOND_DIM_INDEX);
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

static inline aclnnStatus CheckFp4Shape(const QuantMatmulActivationQuantParams& params,
                                        const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    if (shapeInfo.kDim <= 2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "K", std::to_string(shapeInfo.kDim).c_str(),
            "when the dtypes of x1, x2 and y are FP4, the K dimension must be greater than 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t x1InnerAxis = params.transposeX1 ? shapeInfo.mDim : shapeInfo.kDim;
    if (x1InnerAxis % FP4_PACK_RATIO != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, params.transposeX1 ? "x1 M" : "x1 K", std::to_string(x1InnerAxis).c_str(),
            "when the dtypes of x1, x2 and y are FP4, the inner axis of x1 must be even");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t x2InnerAxis = params.transposeX2 ? shapeInfo.kDim : shapeInfo.nDim;
    if (x2InnerAxis % FP4_PACK_RATIO != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, params.transposeX2 ? "x2 K" : "x2 N", std::to_string(x2InnerAxis).c_str(),
            "when the dtypes of x1, x2 and y are FP4, the inner axis of x2 must be even");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckMxScaleLastDim(const QuantMatmulActivationQuantParams& params, const char* apiName)
{
    auto scale1LastDimValue = params.x1Scale->GetViewShape().GetDim(params.x1Scale->GetViewShape().GetDimNum() - 1);
    auto scale2LastDimValue = params.x2Scale->GetViewShape().GetDim(params.x2Scale->GetViewShape().GetDimNum() - 1);
    if (scale1LastDimValue != MX_SCALE_PAIR_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x1Scale", op::ToString(params.x1Scale->GetViewShape()).GetString(),
            FormatString("when the quantization mode is mx, the last dimension of x1Scale must be %lld",
                         static_cast<long long>(MX_SCALE_PAIR_SIZE))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale2LastDimValue != MX_SCALE_PAIR_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "x2Scale", op::ToString(params.x2Scale->GetViewShape()).GetString(),
            FormatString("when the quantization mode is mx, the last dimension of x2Scale must be %lld",
                         static_cast<long long>(MX_SCALE_PAIR_SIZE))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInputScaleShapes(const QuantMatmulActivationQuantParams& params,
                                                const MatmulShapeInfo& shapeInfo, const char* apiName)
{
    if (InferOutputBatchSize(params, apiName) == BATCH_INFER_FAILED) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShapes(params, shapeInfo, x1ScaleExpectShape, x2ScaleExpectShape);

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

static inline aclnnStatus CheckOutputShape(const QuantMatmulActivationQuantParams& params,
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
    if (IsSwiGluActivation(params) && (shapeInfo.nDim <= 0 || shapeInfo.nDim % MX_SCALE_STORAGE_GROUP != 0)) {
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
    int64_t expectedScaleN = Ops::Base::CeilDiv(outputN, MX_SCALE_STORAGE_GROUP);
    if (yScaleView.GetDim(yScaleDimNum - LAST_THIRD_DIM_INDEX) != shapeInfo.mDim ||
        yScaleView.GetDim(yScaleDimNum - LAST_SECOND_DIM_INDEX) != expectedScaleN ||
        yScaleView.GetDim(yScaleDimNum - LAST_FIRST_DIM_INDEX) != MX_SCALE_PAIR_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            apiName, "yScale", op::ToString(yScaleView).GetString(),
            FormatString("yScale last three dims must be [%lld, %lld, %lld], but got [%lld, %lld, %lld]",
                         static_cast<long long>(shapeInfo.mDim), static_cast<long long>(expectedScaleN),
                         static_cast<long long>(MX_SCALE_PAIR_SIZE),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_THIRD_DIM_INDEX)),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_SECOND_DIM_INDEX)),
                         static_cast<long long>(yScaleView.GetDim(yScaleDimNum - LAST_FIRST_DIM_INDEX)))
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline bool CheckGroupSize(QuantMatmulActivationQuantParams& params, const char* apiName)
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
        params.groupSize = (uint64_t{1} << GROUP_M_OFFSET) | (uint64_t{1} << GROUP_N_OFFSET) |
                           static_cast<uint64_t>(PERGROUP_GROUP_SIZE);
    } else if ((static_cast<uint64_t>(groupSize) >> GROUP_RESERVED_BIT_OFFSET) != 0 ||
               groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1UL || groupSizeN != 1UL) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%lld, %llu, %llu, %llu", static_cast<long long>(groupSize),
                         static_cast<unsigned long long>(groupSizeM), static_cast<unsigned long long>(groupSizeN),
                         static_cast<unsigned long long>(groupSizeK))
                .c_str(),
            "when the quantization mode is mx, groupSize must be 0 or 4295032864: "
            "bits [32,47], [16,31] and [0,15] encode groupSizeM=1, groupSizeN=1 and groupSizeK=32; "
            "bits [48,63] must be zero");
        return false;
    }

    OP_LOGD("Group-size validation succeeded.");
    return true;
}

// Preserve caller descriptors; all preprocessing metadata belongs to executor-owned views.
static inline bool PrepareInputViews(QuantMatmulActivationQuantParams& params, aclOpExecutor* executor)
{
    CHECK_RET(executor != nullptr, false);
    for (auto* input : {&params.x1, &params.x2, &params.x1Scale, &params.x2Scale, &params.bias, &params.x3}) {
        if (*input == nullptr) {
            continue;
        }
        // Clone the base-storage descriptor: Contiguous validates against the original capacity and applies
        // the offset itself. Its result (or the transpose fast path) has compact storage for kernel launch.
        // Do not create an offset-reduced explicit-stride view and then restore its original storage shape.
        auto view = executor->CreateView(*input, (*input)->GetStorageShape(), 0);
        CHECK_RET(view != nullptr, false);
        view->SetViewShape((*input)->GetViewShape());
        view->SetViewStrides((*input)->GetViewStrides());
        view->SetViewOffset((*input)->GetViewOffset());
        view->SetViewFormat((*input)->GetViewFormat());
        view->SetStorageFormat((*input)->GetStorageFormat());
        view->SetOriginalFormat((*input)->GetOriginalFormat());
        view->SetOriginalShape((*input)->GetViewShape());
        *input = view;
    }
    return true;
}

static inline aclnnStatus MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, bool transposeScale,
                                                   const op::Shape& expectedShape, aclOpExecutor* executor,
                                                   const char* inputNames, const char* apiName)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MIN_RANK) {
        OP_LOGD("MX scale tensor is absent or does not require contiguous conversion.");
        return ACLNN_SUCCESS;
    }
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto& scaleShape = mxScaleTensor->GetViewShape();
    const auto& scaleStrides = mxScaleTensor->GetViewStrides();
    const auto rowAxis = scaleShape.GetDimNum() - LAST_THIRD_DIM_INDEX;
    const auto columnAxis = scaleShape.GetDimNum() - LAST_SECOND_DIM_INDEX;
    const auto pairAxis = scaleShape.GetDimNum() - LAST_FIRST_DIM_INDEX;
    // Compare recognizable matrix-axis layouts before Contiguous can erase their transpose state.
    // Singleton axes are equivalent in either order; other strided layouts keep the existing conversion path.
    if (scaleShape.GetDim(rowAxis) > 1 && scaleShape.GetDim(columnAxis) > 1 &&
        scaleShape.GetDim(pairAxis) == MX_SCALE_PAIR_SIZE && scaleStrides[pairAxis] == 1) {
        const bool scaleTransposeView = scaleStrides[rowAxis] == MX_SCALE_PAIR_SIZE &&
                                        scaleStrides[columnAxis] == MX_SCALE_PAIR_SIZE * scaleShape.GetDim(rowAxis);
        const bool scaleContiguousView = scaleStrides[columnAxis] == MX_SCALE_PAIR_SIZE &&
                                         scaleStrides[rowAxis] == MX_SCALE_PAIR_SIZE * scaleShape.GetDim(columnAxis);
        if ((transposeScale && scaleContiguousView) || (!transposeScale && scaleTransposeView)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName, inputNames,
                FormatString("matrix transpose view=%s, scale transpose view=%s", transposeScale ? "true" : "false",
                             scaleTransposeView ? "true" : "false")
                    .c_str(),
                "the matrix and its MX scale must use matching transpose views");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (scaleShape.GetDim(rowAxis) == 1 || scaleShape.GetDim(columnAxis) == 1) {
        // Swapping a singleton axis preserves element order. Normalize only an equivalent full shape;
        // batch dimensions and the number of scale groups must still match the matrix.
        auto swappedShape = scaleShape;
        swappedShape.SetDim(rowAxis, scaleShape.GetDim(columnAxis));
        swappedShape.SetDim(columnAxis, scaleShape.GetDim(rowAxis));
        if (scaleShape == expectedShape || swappedShape == expectedShape) {
            transposeScale = scaleShape != expectedShape;
        }
    }
    if (transposeScale) {
        // Follow matrix view normalization, or restore the canonical shape for equivalent singleton axes.
        auto shape = mxScaleTensor->GetViewShape();
        auto strides = mxScaleTensor->GetViewStrides();
        const auto rowDim = shape.GetDim(rowAxis);
        shape.SetDim(rowAxis, shape.GetDim(columnAxis));
        shape.SetDim(columnAxis, rowDim);
        std::swap(strides[rowAxis], strides[columnAxis]);
        auto scaleView = executor->CreateView(mxScaleTensor, mxScaleTensor->GetStorageShape(), 0);
        CHECK_RET(scaleView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        scaleView->SetViewShape(shape);
        scaleView->SetViewStrides(strides);
        scaleView->SetViewOffset(mxScaleTensor->GetViewOffset());
        scaleView->SetOriginalShape(shape);
        mxScaleTensor = scaleView;
    }
    // Contiguous checks every stride, including the last scale-pair axis, before reusing storage.
    mxScaleTensor = Ops::NN::ContiguousMxScale(mxScaleTensor, executor);
    return mxScaleTensor != nullptr ? ACLNN_SUCCESS : ACLNN_ERR_INNER_NULLPTR;
}

// Prepare scales, bias and the left matrix; ND and WeightNZ handle x2 separately.
static inline aclnnStatus PrepareCommonInputs(QuantMatmulActivationQuantParams& params, aclOpExecutor* executor)
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

    if (params.x3 != nullptr) {
        // Torch labels dense rank-3/4/5 inputs as NCL/NCHW/NCDHW; the kernel consumes ND.
        params.x3 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x3);
        CHECK_RET(params.x3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        params.x3 = l0op::Contiguous(params.x3, executor);
        CHECK_RET(params.x3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto reformattedX1 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x1);
    CHECK_RET(reformattedX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x1 = reformattedX1;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.x1, params.transposeX1, executor),
              ACLNN_ERR_INNER_NULLPTR);
    if (params.bias != nullptr) {
        params.bias = l0op::Contiguous(params.bias, executor);
        CHECK_RET(params.bias != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus AddQuantMatmulActivationQuantTasks(const QuantMatmulActivationQuantParams& params,
                                                             aclOpExecutor* executor)
{
    auto [yResult, yScaleResult] = l0op::QuantMatmulActivationQuant(
        params.x1, params.x2, params.bias, params.x1Scale, params.x2Scale, params.x3, params.transposeX1,
        params.transposeX2, params.groupSize, params.activationType, params.yDtype, params.quantMode, params.roundMode,
        params.scaleAlg, params.dstTypeMax, executor);
    CHECK_RET(yResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(yScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(l0op::ViewCopy(yResult, params.y, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(yScaleResult, params.yScale, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace QBMMActivationQuant
#endif
