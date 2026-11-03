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

constexpr int64_t kDstTypeFloat16 = 1;
constexpr int64_t kDstTypeBFloat16 = 27;
constexpr int64_t kFp4InByte = 2;
constexpr int64_t kHif4BlockSize = 64;
constexpr int64_t kXDimNumMin = 2;
constexpr int64_t kXDimNumMax = 7;
constexpr int64_t kScaleDimNumMin = 3;
constexpr int64_t kScaleDimNumMax = 8;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckDstType(int64_t dstType)
{
    TORCH_CHECK(dstType == kDstTypeFloat16 || dstType == kDstTypeBFloat16,
                "dst_type should be 1 (float16) or 27 (bfloat16), but got ", dstType);
}

c10::SmallVector<int64_t, op_infer::SIZE> GetYShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() >= kXDimNumMin, "x rank should be greater than or equal to 2");
    auto shape = op_infer::array_to_small_vector(x.sizes());
    shape[x.dim() - 1] = x.size(x.dim() - 1) * kFp4InByte;
    return shape;
}

at::Tensor anti_hi_f4_quant(const at::Tensor& x, const at::Tensor& scale, int64_t dst_type)
{
    CheckNpuTensor(x, "x");
    CheckNpuTensor(scale, "scale");
    CheckDstType(dst_type);
    TORCH_CHECK(x.dim() >= kXDimNumMin && x.dim() <= kXDimNumMax, "x rank should be in [2, 7]");
    TORCH_CHECK(scale.dim() >= kScaleDimNumMin && scale.dim() <= kScaleDimNumMax, "scale rank should be in [3, 8]");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte, "x dtype should be uint8 (packed HiFloat4)");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
                "scale dtype should be float32 (packed HiFloat4_SCALE storage)");
    TORCH_CHECK(x.device() == scale.device(), "x and scale should be on the same device");
    TORCH_CHECK(scale.is_contiguous(), "scale should be contiguous");
    TORCH_CHECK(x.size(-1) % (kHif4BlockSize / kFp4InByte) == 0 && x.size(-2) % kHif4BlockSize == 0,
                "x logical last and second last dimensions should be multiples of 64");

    const auto yShape = GetYShape(x);
    const auto yScalarType = (dst_type == kDstTypeFloat16) ? at::ScalarType::Half : at::ScalarType::BFloat16;
    at::Tensor y = at::empty(yShape, x.options().dtype(yScalarType));

    // 输入按实际 acl 数据类型转发：x 为 HIFLOAT4（torch 侧 uint8 打包存储），scale 为 HIFLOAT4_SCALE。
    TensorWrapper xWrapper{x, ACL_HIFLOAT4_NUM};
    TensorWrapper scaleWrapper{scale, ACL_HIFLOAT4_SCALE_NUM};
    ACLNN_CMD(aclnnAntiHiF4Quant, xWrapper, scaleWrapper, dst_type, y);
    return y;
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("anti_hi_f4_quant", &cann_ops_nn::quant::anti_hi_f4_quant, "AntiHiF4Quant on NPU");
}
