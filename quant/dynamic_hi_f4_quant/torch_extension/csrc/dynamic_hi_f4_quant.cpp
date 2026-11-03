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
namespace quant {

constexpr int64_t kHif4BlockSize = 64;
constexpr int64_t kFp4InByte = 2;
constexpr int64_t kScaleTrailingDim = 1;
// scale_alg: 0-标准阈值(4/2), 1-放宽阈值(3.75/1.875)
constexpr int64_t kScaleAlgDefault = 0;
constexpr int64_t kScaleAlgRelaxed = 1;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    TORCH_CHECK(factor > 0, "factor must be positive");
    return (value + factor - 1) / factor;
}

void CheckAxis(const at::Tensor& x, int64_t axis)
{
    TORCH_CHECK(axis >= -1 * x.dim() && axis < x.dim(), "axis is out of input dimension range [", -1 * x.dim(), ", ",
                x.dim(), "), but got ", axis);
    const int64_t axisIdx = axis < 0 ? axis + x.dim() : axis;
    TORCH_CHECK(axisIdx == x.dim() - 1 || axisIdx == x.dim() - 2,
                "axis should select the last or second last dimension");
    TORCH_CHECK(x.size(-1) % kHif4BlockSize == 0, "x last dimension should be a multiple of 64");
    if (axisIdx == x.dim() - 2) {
        TORCH_CHECK(x.size(-2) % kHif4BlockSize == 0, "x second last dimension should be a multiple of 64");
    }
}

void ValidateScaleAlg(int64_t scaleAlg)
{
    TORCH_CHECK(scaleAlg == kScaleAlgDefault || scaleAlg == kScaleAlgRelaxed, "scale_alg should be 0 or 1, but got ",
                scaleAlg);
}

c10::SmallVector<int64_t, op_infer::SIZE> GetYShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= 2, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kFp4InByte);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScaleShape(const at::Tensor& x, int64_t axis)
{
    TORCH_CHECK(x.dim() >= 2, "x rank should be greater than or equal to 2");
    const int64_t axisIdx = axis < 0 ? axis + x.dim() : axis;
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[axisIdx] = CeilDiv(x.size(axisIdx), kHif4BlockSize);
    shape.emplace_back(kScaleTrailingDim);
    return shape;
}

std::tuple<at::Tensor, at::Tensor> dynamic_hi_f4_quant(const at::Tensor& x, int64_t axis, const int64_t scale_alg)
{
    CheckNpuTensor(x, "x");
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 7, "x rank should be in [2, 7]");
    CheckAxis(x, axis);
    ValidateScaleAlg(scale_alg);
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16,
                "x dtype should be float16 or bfloat16");

    const auto yShape = GetYShape(x);
    at::Tensor y = at::empty(yShape, x.options().dtype(at::ScalarType::Byte));

    const auto scaleShape = GetScaleShape(x, axis);
    at::Tensor yScale = at::empty(scaleShape, x.options().dtype(at::ScalarType::Float));

    // 输出按实际 acl 数据类型转发：y 为 HIFLOAT4，scale 为 HIFLOAT4_SCALE。
    TensorWrapper yWrapper{y, ACL_HIFLOAT4_NUM};
    TensorWrapper yScaleWrapper{yScale, ACL_HIFLOAT4_SCALE_NUM};
    ACLNN_CMD(aclnnDynamicHiF4Quant, x, axis, scale_alg, yWrapper, yScaleWrapper);
    return std::make_tuple(y, yScale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dynamic_hi_f4_quant", &cann_ops_nn::quant::dynamic_hi_f4_quant, "DynamicHiF4Quant on NPU");
}
