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
namespace norm {

constexpr int64_t kHif4BlockSize = 64;
constexpr int64_t kFp4InByte = 2;
constexpr int64_t kMinRank = 2;
constexpr int64_t kScaleTrailingDim = 1;
constexpr int64_t kRstdTrailingDim = 1;

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

void CheckScaleAlg(int64_t scale_alg)
{
    TORCH_CHECK(scale_alg == 0 || scale_alg == 1, "scale_alg should be 0 (mse_first) or 1 (unbiased), but got ",
                scale_alg);
}

c10::SmallVector<int64_t, op_infer::SIZE> GetYShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= kMinRank, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kFp4InByte);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScaleShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= kMinRank, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kHif4BlockSize);
    shape.emplace_back(kScaleTrailingDim);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetRstdShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= kMinRank, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = kRstdTrailingDim;
    return shape;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> add_rms_norm_hi_f4_quant(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    double epsilon, bool output_rstd, int64_t scale_alg)
{
    CheckNpuTensor(x1, "x1");
    CheckNpuTensor(x2, "x2");
    CheckNpuTensor(gamma, "gamma");
    CheckScaleAlg(scale_alg);
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "x2 must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x1.scalar_type() == at::ScalarType::Half || x1.scalar_type() == at::ScalarType::BFloat16,
                "x1 dtype should be float16 or bfloat16, but got ", x1.scalar_type());
    TORCH_CHECK(x2.sizes() == x1.sizes(), "x2 shape must match x1 (", x1.sizes(), "), but got ", x2.sizes());
    TORCH_CHECK(x2.scalar_type() == x1.scalar_type(), "x2 dtype must match x1, but got ", x2.scalar_type());
    TORCH_CHECK(gamma.dim() == 1, "gamma should be 1-D, but got ", gamma.dim(), "-D");
    TORCH_CHECK(gamma.size(0) == x1.size(x1.dim() - 1), "gamma dim must match x1 last dim (", x1.size(x1.dim() - 1),
                "), but got ", gamma.size(0));
    TORCH_CHECK(x1.size(x1.dim() - 1) % kHif4BlockSize == 0, "x1 last dim must be a multiple of ", kHif4BlockSize,
                ", but got ", x1.size(x1.dim() - 1));
    if (beta.has_value()) {
        const at::Tensor& betaTensor = beta.value();
        CheckNpuTensor(betaTensor, "beta");
        TORCH_CHECK(betaTensor.is_contiguous(), "beta must be contiguous");
        TORCH_CHECK(betaTensor.dim() == 1 && betaTensor.size(0) == gamma.size(0),
                    "beta shape must match gamma, but got ", betaTensor.sizes());
    }

    const auto yShape = GetYShape(x1);
    at::Tensor y = at::empty(yShape, x1.options().dtype(at::ScalarType::Byte));

    // xOut 为 x1 与 x2 的逐元素和，shape 与 dtype 均与 x1 一致。
    at::Tensor xOut = at::empty_like(x1);

    const auto scaleShape = GetScaleShape(x1);
    at::Tensor scaleOut = at::empty(scaleShape, x1.options().dtype(at::ScalarType::Float));

    const auto rstdShape = GetRstdShape(x1);
    // rstd 恒为有效输出（全零初始化）：output_rstd 为 false 时算子不写 rstd，保持全零。
    at::Tensor rstd = at::zeros(rstdShape, x1.options().dtype(at::ScalarType::Float));

    // 输出按实际 acl 数据类型转发：y 为 HIFLOAT4，x 与 x1 同 dtype，scale 为 HIFLOAT4_SCALE，rstd 为 FLOAT32。
    TensorWrapper yWrapper{y, ACL_HIFLOAT4};
    TensorWrapper xWrapper{
        xOut, static_cast<aclDataType>(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(x1.scalar_type())])};
    TensorWrapper scaleWrapper{scaleOut, ACL_HIFLOAT4_SCALE};
    TensorWrapper rstdWrapper{rstd, ACL_FLOAT};
    ACLNN_CMD(aclnnAddRmsNormHiF4Quant, x1, x2, gamma, beta, epsilon, output_rstd, scale_alg, yWrapper, xWrapper,
              scaleWrapper, rstdWrapper);
    return std::make_tuple(y, xOut, scaleOut, rstd);
}

} // namespace norm
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_rms_norm_hi_f4_quant", &cann_ops_nn::norm::add_rms_norm_hi_f4_quant, "AddRmsNormHiF4Quant on NPU");
}
