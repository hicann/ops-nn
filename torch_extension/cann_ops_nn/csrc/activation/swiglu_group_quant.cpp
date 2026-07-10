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
#include "../common/aclnn_common.h"

namespace cann_ops_nn {
namespace activation {
namespace {
constexpr int64_t kSplitFactor = 2;
constexpr int64_t kBlockFp8QuantMode = 0;
constexpr int64_t kMxQuantMode = 1;
constexpr int64_t kStaticHifp8QuantMode = 2;
constexpr int64_t kDynamicHifp8QuantMode = 3;
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

int64_t CeilDiv(int64_t value, int64_t factor)
{
    TORCH_CHECK(factor > 0, "factor must be positive");
    return (value + factor - 1) / factor;
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

c10::SmallVector<int64_t, op_infer::SIZE> GetSwigluShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    const int64_t lastDim = x.size(x.dim() - 1);
    TORCH_CHECK(lastDim % kSplitFactor == 0, "x last dim size should be even");

    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = lastDim / kSplitFactor;
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetQuantOutputShape(const at::Tensor& x, aclDataType yAclType,
                                                              int64_t quantMode)
{
    auto yShape = GetSwigluShape(x);
    if (quantMode == kMxQuantMode && IsFp4Dtype(yAclType)) {
        yShape[x.dim() - 1] = CeilDiv(yShape[x.dim() - 1], kFp4InByte);
    }
    return yShape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScaleShape(const at::Tensor& x,
                                                        const c10::optional<at::Tensor>& groupIndex, int64_t quantMode)
{
    const int64_t swigluLastDim = x.size(x.dim() - 1) / kSplitFactor;
    c10::SmallVector<int64_t, op_infer::SIZE> scaleShape;

    if (quantMode == kStaticHifp8QuantMode) {
        scaleShape.emplace_back(0);
        return scaleShape;
    }
    if (quantMode == kDynamicHifp8QuantMode) {
        if (groupIndex.has_value() && groupIndex.value().defined()) {
            return op_infer::array_to_small_vector(groupIndex.value().sizes());
        }
        scaleShape.emplace_back(1);
        return scaleShape;
    }

    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        scaleShape.emplace_back(x.size(i));
    }
    if (quantMode == kMxQuantMode) {
        int64_t tailDim = CeilDiv(swigluLastDim, kMxBlockSize);
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
    const c10::optional<at::Tensor>& scale, int64_t dst_type, int64_t quant_mode, int64_t block_size, bool round_scale,
    double clamp_limit, double dst_type_max, bool output_origin)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(group_index, "group_index");
    CheckOptionalNpuTensor(scale, "scale");
    TORCH_CHECK(quant_mode >= kBlockFp8QuantMode && quant_mode <= kDynamicHifp8QuantMode,
                "quant_mode should be 0, 1, 2 or 3, but got ", quant_mode);

    const aclDataType yAclType = GetOutputAclType(dst_type, quant_mode);
    at::Tensor y = at::empty(GetQuantOutputShape(x, yAclType, quant_mode),
                             x.options().dtype(GetQuantOutputScalarType(yAclType)));

    const bool isMxQuant = quant_mode == kMxQuantMode;
    const aclDataType yScaleAclType = isMxQuant ? ACL_FLOAT8_E8M0 : ACL_FLOAT;
    const at::ScalarType yScaleScalarType = isMxQuant ? GetFp8ScalarType(ACL_FLOAT8_E8M0) : at::ScalarType::Float;
    at::Tensor yScale = at::empty(GetScaleShape(x, group_index, quant_mode), x.options().dtype(yScaleScalarType));

    at::Tensor yOrigin = at::empty({0}, x.options());
    if (output_origin) {
        yOrigin = at::empty(GetSwigluShape(x), x.options());
    }

    TensorWrapper yWrapper{y, yAclType};
    TensorWrapper yScaleWrapper{yScale, yScaleAclType};
    TensorWrapper yOriginWrapper{yOrigin, ConvertToAclDataType(yOrigin.scalar_type())};
    ACLNN_CMD(aclnnSwigluGroupQuant, x, weight, group_index, scale, yAclType, quant_mode, block_size, round_scale,
              clamp_limit, dst_type_max, output_origin, yWrapper, yScaleWrapper, yOriginWrapper);
    return std::make_tuple(y, yScale, yOrigin);
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group_quant", &cann_ops_nn::activation::swiglu_group_quant, "SwigluGroupQuant on NPU");
}
