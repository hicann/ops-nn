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

void CheckRoundMode(const std::string& roundMode)
{
    TORCH_CHECK(roundMode == "rint" || roundMode == "trunc", "round_mode should be one of 'rint' or 'trunc', but got ",
                roundMode);
}

// scale_alg: 0-标准阈值(4/2), 1-放宽阈值(3.75/1.875)
constexpr int64_t kScaleAlgDefault = 0;
constexpr int64_t kScaleAlgRelaxed = 1;

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

c10::SmallVector<int64_t, op_infer::SIZE> GetScale1Shape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= 2, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = CeilDiv(x.size(x.dim() - 1), kHif4BlockSize);
    shape.emplace_back(kScaleTrailingDim);
    return shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScale2Shape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= 2, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 2] = CeilDiv(x.size(x.dim() - 2), kHif4BlockSize);
    shape.emplace_back(kScaleTrailingDim);
    return shape;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dynamic_block_hi_f4_quant(const at::Tensor& x,
                                                                         const std::string& round_mode,
                                                                         const int64_t scale_alg)
{
    CheckNpuTensor(x, "x");
    CheckRoundMode(round_mode);
    ValidateScaleAlg(scale_alg);
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 7, "x rank should be in [2, 7]");
    TORCH_CHECK(x.size(-1) % kHif4BlockSize == 0 && x.size(-2) % kHif4BlockSize == 0,
                "x last and second last dimensions should be multiples of 64");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16,
                "x dtype should be float16 or bfloat16");

    const auto yShape = GetYShape(x);
    at::Tensor y = at::empty(yShape, x.options().dtype(at::ScalarType::Byte));

    const auto scale1Shape = GetScale1Shape(x);
    const auto scale2Shape = GetScale2Shape(x);
    at::Tensor yScale1 = at::empty(scale1Shape, x.options().dtype(at::ScalarType::Float));
    at::Tensor yScale2 = at::empty(scale2Shape, x.options().dtype(at::ScalarType::Float));

    // 输出按实际 acl 数据类型转发：y 为 HIFLOAT4，scale1/scale2 为 HIFLOAT4_SCALE。
    // y 显式指定 packedDim=最后一维：acl 侧按元素级 shape（最后一维×2，与 x 一致）声明，
    // 满足算子 tiling 的 y.shape == x.shape 校验；torch 侧 y 保持 packed（D/2）视图。
    TensorWrapper yWrapper{y, ACL_HIFLOAT4_NUM, y.dim() - 1};
    TensorWrapper yScale1Wrapper{yScale1, ACL_HIFLOAT4_SCALE_NUM};
    TensorWrapper yScale2Wrapper{yScale2, ACL_HIFLOAT4_SCALE_NUM};
    ACLNN_CMD(aclnnDynamicBlockHiF4Quant, x, round_mode, scale_alg, yWrapper, yScale1Wrapper, yScale2Wrapper);
    return std::make_tuple(y, yScale1, yScale2);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dynamic_block_hi_f4_quant", &cann_ops_nn::quant::dynamic_block_hi_f4_quant, "DynamicBlockHiF4Quant on NPU");
}
