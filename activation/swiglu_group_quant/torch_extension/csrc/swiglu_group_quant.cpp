/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace activation {
namespace {
constexpr int64_t kSplitFactor = 2;
constexpr int64_t kBlockFp8QuantMode = 0;
constexpr int64_t kMxQuantMode = 1;
constexpr int64_t kStaticHifp8QuantMode = 2;
constexpr int64_t kDynamicHifp8QuantMode = 3;
constexpr int64_t kMxQuantV2Mode = 5;
constexpr int64_t kBlockFp8BlockSize = 128;
constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kMxScaleAlign = 2;
constexpr int64_t kFp4InByte = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
}

c10::SymInt CeilDiv(const c10::SymInt& value, int64_t factor)
{
    const c10::SymInt divisor(factor);
    const c10::SymInt remainder = value % divisor;
    return value / divisor + (remainder + c10::SymInt(factor - 1)) / divisor;
}

bool IsFp4Dtype(aclDataType dtype) { return dtype == ACL_FLOAT4_E2M1 || dtype == ACL_FLOAT4_E1M2; }

bool IsFp8Dtype(aclDataType dtype) { return dtype == ACL_FLOAT8_E5M2 || dtype == ACL_FLOAT8_E4M3FN; }

at::ScalarType GetFp8ScalarType(aclDataType dtype)
{
    if (dtype == ACL_FLOAT8_E5M2) {
        return at::ScalarType::Float8_e5m2;
    }
    if (dtype == ACL_FLOAT8_E4M3FN) {
        return at::ScalarType::Float8_e4m3fn;
    }
    if (dtype == ACL_FLOAT8_E8M0) {
        return at::ScalarType::Float8_e8m0fnu;
    }
    TORCH_CHECK(false, "unsupported fp8 dtype: ", static_cast<int64_t>(dtype));
}

at::ScalarType GetQuantOutputScalarType(aclDataType dtype)
{
    return IsFp8Dtype(dtype) ? GetFp8ScalarType(dtype) : at::ScalarType::Byte;
}

bool IsSupportedOutputDtype(aclDataType dtype)
{
    return IsFp8Dtype(dtype) || IsFp4Dtype(dtype) || dtype == ACL_HIFLOAT8;
}

bool IsHifp8QuantMode(int64_t quantMode)
{
    return quantMode == kStaticHifp8QuantMode || quantMode == kDynamicHifp8QuantMode;
}

c10::SymDimVector GetSwigluShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    const c10::SymInt lastDim = x.sym_size(x.dim() - 1);

    c10::SymDimVector shape(x.sym_sizes());
    shape[x.dim() - 1] = lastDim / kSplitFactor;
    return shape;
}

c10::SymDimVector GetQuantOutputShape(const at::Tensor& x, aclDataType yAclType, int64_t quantMode)
{
    auto yShape = GetSwigluShape(x);
    if (quantMode == kMxQuantMode && IsFp4Dtype(yAclType)) {
        yShape[x.dim() - 1] = CeilDiv(yShape[x.dim() - 1], kFp4InByte);
    }
    return yShape;
}

c10::SymDimVector GetScaleShape(const at::Tensor& x, const c10::optional<at::Tensor>& groupIndex, int64_t quantMode)
{
    const c10::SymInt swigluLastDim = x.sym_size(x.dim() - 1) / kSplitFactor;
    c10::SymDimVector scaleShape;

    if (quantMode == kStaticHifp8QuantMode) {
        scaleShape.emplace_back(0);
        return scaleShape;
    }
    if (quantMode == kDynamicHifp8QuantMode) {
        if (groupIndex.has_value() && groupIndex.value().defined()) {
            return c10::SymDimVector(groupIndex.value().sym_sizes());
        }
        scaleShape.emplace_back(1);
        return scaleShape;
    }

    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        scaleShape.emplace_back(x.sym_size(i));
    }
    if (quantMode == kMxQuantMode || quantMode == kMxQuantV2Mode) {
        c10::SymInt tailDim = CeilDiv(swigluLastDim, kMxBlockSize);
        tailDim = CeilDiv(tailDim, kMxScaleAlign);
        scaleShape.emplace_back(tailDim);
        scaleShape.emplace_back(kMxScaleAlign);
    } else {
        scaleShape.emplace_back(CeilDiv(swigluLastDim, kBlockFp8BlockSize));
    }
    return scaleShape;
}

aclDataType GetOutputAclType(int64_t dstType, int64_t quantMode)
{
    if (IsHifp8QuantMode(quantMode)) {
        return ACL_HIFLOAT8;
    }

    aclDataType yAclType = GetAclDataType(dstType);
    TORCH_CHECK(IsSupportedOutputDtype(yAclType), "unsupported dst_type: ", dstType);
    TORCH_CHECK(quantMode == kMxQuantMode || !IsFp4Dtype(yAclType),
                "dst_type FLOAT4_E2M1/FLOAT4_E1M2 requires quant_mode=1");
    return yAclType;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& group_index,
    const c10::optional<at::Tensor>& scale, int64_t dst_type, const c10::optional<int64_t>& quant_mode,
    const c10::optional<int64_t>& block_size, const c10::optional<bool>& round_scale, double clamp_limit,
    double dst_type_max, bool output_origin, double alpha, double bias)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(group_index, "group_index");
    CheckOptionalNpuTensor(scale, "scale");

    const double resolvedAlpha = alpha;
    const double resolvedBias = bias;

    const int64_t resolvedQuantMode = quant_mode.value_or(kBlockFp8QuantMode);
    const int64_t resolvedBlockSize = block_size.value_or(0);
    const bool resolvedRoundScale = round_scale.value_or(false);
    const bool useV2 = resolvedQuantMode == kMxQuantV2Mode;

    // The legacy ACLNN entry has no alpha/bias arguments; do not silently discard them.
    TORCH_CHECK(useV2 || (resolvedAlpha == 1.0 && resolvedBias == 0.0),
                "alpha/bias only support quant_mode=5, got quant_mode ", resolvedQuantMode);

    const aclDataType yAclType = GetOutputAclType(dst_type, resolvedQuantMode);
    at::Tensor y = at::empty_symint(GetQuantOutputShape(x, yAclType, resolvedQuantMode),
                                    x.options().dtype(GetQuantOutputScalarType(yAclType)));

    const bool isMxQuant = resolvedQuantMode == kMxQuantMode || resolvedQuantMode == kMxQuantV2Mode;
    const aclDataType yScaleAclType = isMxQuant ? ACL_FLOAT8_E8M0 : ACL_FLOAT;
    const at::ScalarType yScaleScalarType = isMxQuant ? GetFp8ScalarType(ACL_FLOAT8_E8M0) : at::ScalarType::Float;
    at::Tensor yScale = at::empty_symint(GetScaleShape(x, group_index, resolvedQuantMode),
                                         x.options().dtype(yScaleScalarType));

    const c10::SymDimVector emptyShape{0};
    at::Tensor yOrigin = at::empty_symint(emptyShape, x.options());
    if (output_origin) {
        yOrigin = at::empty_symint(GetSwigluShape(x), x.options());
    }

    TensorWrapper yWrapper{y, yAclType};
    TensorWrapper yScaleWrapper{yScale, yScaleAclType};
    TensorWrapper yOriginWrapper{yOrigin, ConvertToAclDataType(yOrigin.scalar_type())};
    if (useV2) {
        ACLNN_CMD(aclnnSwigluGroupQuantV2, x, weight, group_index, scale, yAclType, resolvedQuantMode,
                  resolvedBlockSize, resolvedRoundScale, clamp_limit, dst_type_max, output_origin, resolvedAlpha,
                  resolvedBias, yWrapper, yScaleWrapper, yOriginWrapper);
    } else {
        ACLNN_CMD(aclnnSwigluGroupQuant, x, weight, group_index, scale, yAclType, resolvedQuantMode, resolvedBlockSize,
                  resolvedRoundScale, clamp_limit, dst_type_max, output_origin, yWrapper, yScaleWrapper,
                  yOriginWrapper);
    }
    return std::make_tuple(y, yScale, yOrigin);
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group_quant", &cann_ops_nn::activation::swiglu_group_quant, "SwigluGroupQuant on NPU");
}
