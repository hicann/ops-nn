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
    TORCH_CHECK(x.dim() >= 1, "x rank should be greater than or equal to 1");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kFp4InByte);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScaleShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= 1, "x rank should be greater than or equal to 1");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kHif4BlockSize);
    shape.emplace_back(kScaleTrailingDim);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetRstdShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= 1, "x rank should be greater than or equal to 1");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = kRstdTrailingDim;
    return shape;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> rms_norm_hi_f4_quant(const at::Tensor& x, const at::Tensor& gamma,
                                                                    const c10::optional<at::Tensor>& beta,
                                                                    double epsilon, bool output_rstd, int64_t scale_alg)
{
    CheckNpuTensor(x, "x");
    CheckNpuTensor(gamma, "gamma");
    CheckScaleAlg(scale_alg);
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16,
                "x dtype should be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(gamma.dim() == 1, "gamma should be 1-D, but got ", gamma.dim(), "-D");
    TORCH_CHECK(gamma.size(0) == x.size(x.dim() - 1), "gamma dim must match x last dim (", x.size(x.dim() - 1),
                "), but got ", gamma.size(0));
    TORCH_CHECK(x.size(x.dim() - 1) % kHif4BlockSize == 0, "x last dim must be a multiple of ", kHif4BlockSize,
                ", but got ", x.size(x.dim() - 1));
    if (beta.has_value()) {
        const at::Tensor& betaTensor = beta.value();
        CheckNpuTensor(betaTensor, "beta");
        TORCH_CHECK(betaTensor.is_contiguous(), "beta must be contiguous");
        TORCH_CHECK(betaTensor.dim() == 1 && betaTensor.size(0) == gamma.size(0),
                    "beta shape must match gamma, but got ", betaTensor.sizes());
    }

    const auto yShape = GetYShape(x);
    at::Tensor y = at::empty(yShape, x.options().dtype(at::ScalarType::Byte));

    const auto scaleShape = GetScaleShape(x);
    at::Tensor scaleOut = at::empty(scaleShape, x.options().dtype(at::ScalarType::Float));

    const auto rstdShape = GetRstdShape(x);
    // rstd 恒为有效输出（全零初始化）：output_rstd 为 false 时算子不写 rstd，保持全零。
    at::Tensor rstd = at::zeros(rstdShape, x.options().dtype(at::ScalarType::Float));

    // 输出按实际 acl 数据类型转发：y 为 HIFLOAT4，scale 为 HIFLOAT4_SCALE，rstd 为 FLOAT32。
    // 当 output_rstd 为 false 时 rstd 由算子填充为全零，仍为有效输出。
    TensorWrapper yWrapper{y, ACL_HIFLOAT4};
    TensorWrapper scaleWrapper{scaleOut, ACL_HIFLOAT4_SCALE};
    TensorWrapper rstdWrapper{rstd, ACL_FLOAT};
    ACLNN_CMD(aclnnRmsNormHiF4Quant, x, gamma, beta, epsilon, output_rstd, scale_alg, yWrapper, scaleWrapper,
              rstdWrapper);
    return std::make_tuple(y, scaleOut, rstd);
}

} // namespace norm
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm_hi_f4_quant", &cann_ops_nn::norm::rms_norm_hi_f4_quant, "RmsNormHiF4Quant on NPU");
}
