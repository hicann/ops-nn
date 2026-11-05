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

#include <cstdint>
#include <limits>
#include <vector>

#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {
namespace {
constexpr int64_t kExpectedDim = 3;
constexpr size_t kPermSize = 3;
constexpr int64_t kDefaultBatchSplitFactor = 1;
constexpr int8_t kKeepDtype = 0;
constexpr int8_t kUseHf32 = 3;

// Must stay in sync with _DEFAULT_PERM_X1/_X2/_Y in ../_perm_defaults.py.
const std::vector<int64_t> kDefaultPermX1 = {1, 0, 2};
const std::vector<int64_t> kDefaultPermX2 = {0, 1, 2};
const std::vector<int64_t> kDefaultPermY = {1, 0, 2};

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
    TORCH_CHECK(tensor.dim() == kExpectedDim, name, " must be 3-dimensional, but got ", tensor.dim());
}

std::vector<int64_t> ResolvePerm(const c10::optional<std::vector<int64_t>>& perm,
                                 const std::vector<int64_t>& default_perm, const char* name)
{
    std::vector<int64_t> result = perm.has_value() ? perm.value() : default_perm;
    TORCH_CHECK(result.size() == kPermSize, name, " must have exactly ", kPermSize, " elements");
    return result;
}

void CheckPerm(const std::vector<int64_t>& perm, const std::vector<std::vector<int64_t>>& supported, const char* name)
{
    for (const auto& candidate : supported) {
        if (perm == candidate) {
            return;
        }
    }
    TORCH_CHECK(false, name, " has an unsupported permutation");
}

void CheckDtype(const at::Tensor& x1, const at::Tensor& x2, const c10::optional<at::Tensor>& scale)
{
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "x1 and x2 must have the same dtype, but got ", x1.scalar_type(),
                " and ", x2.scalar_type());
    TORCH_CHECK(x1.scalar_type() == at::kHalf || x1.scalar_type() == at::kBFloat16 || x1.scalar_type() == at::kFloat,
                "x1/x2 dtype must be float16, bfloat16 or float32, but got ", x1.scalar_type());
    if (scale.has_value() && scale->defined()) {
        TORCH_CHECK(scale->scalar_type() == at::kLong || scale->scalar_type() == at::kUInt64,
                    "scale dtype must be int64 or uint64, but got ", scale->scalar_type());
        TORCH_CHECK(x1.scalar_type() == at::kHalf,
                    "scale is only supported for float16 input with int8 output, but got x1 dtype ", x1.scalar_type());
    }
}

std::vector<int64_t> InferOutputShape(const at::Tensor& x1, const at::Tensor& x2, const std::vector<int64_t>& perm_x1,
                                      const std::vector<int64_t>& perm_x2, bool has_scale, int64_t batch_split_factor)
{
    const int64_t m_dim = x1.size(perm_x1[1]);
    const int64_t batch_dim = x1.size(perm_x1[0]);
    const int64_t n_dim = x2.size(perm_x2[2]);
    int64_t batch_n = 0;
    if (has_scale || batch_split_factor > 1) {
        TORCH_CHECK(n_dim == 0 || batch_dim <= std::numeric_limits<int64_t>::max() / n_dim,
                    "batch * N overflows int64");
        batch_n = batch_dim * n_dim;
    }
    if (has_scale) {
        return {m_dim, 1, batch_n};
    }
    if (batch_split_factor > 1) {
        TORCH_CHECK(batch_dim % batch_split_factor == 0, "batch_split_factor must be a factor of batch, but got ",
                    batch_split_factor);
        return {batch_split_factor, m_dim, batch_n / batch_split_factor};
    }
    return {m_dim, batch_dim, n_dim};
}
} // namespace

at::Tensor transpose_batch_mat_mul(const at::Tensor& x1, const at::Tensor& x2, const c10::optional<at::Tensor>& bias,
                                   const c10::optional<at::Tensor>& scale, c10::optional<std::vector<int64_t>>& perm_x1,
                                   c10::optional<std::vector<int64_t>>& perm_x2,
                                   c10::optional<std::vector<int64_t>>& perm_y, bool enable_hf32,
                                   c10::optional<int64_t> batch_split_factor)
{
    CheckNpuTensor(x1, "x1");
    CheckNpuTensor(x2, "x2");
    CheckDtype(x1, x2, scale);

    const bool has_scale = scale.has_value() && scale->defined();
    const int64_t batch_split_factor_value = batch_split_factor.value_or(kDefaultBatchSplitFactor);
    TORCH_CHECK(batch_split_factor_value > 0, "batch_split_factor must be positive, but got ",
                batch_split_factor_value);

    const std::vector<int64_t> perm_x1_value = ResolvePerm(perm_x1, kDefaultPermX1, "perm_x1");
    const std::vector<int64_t> perm_x2_value = ResolvePerm(perm_x2, kDefaultPermX2, "perm_x2");
    const std::vector<int64_t> perm_y_value = ResolvePerm(perm_y, kDefaultPermY, "perm_y");
    CheckPerm(perm_x1_value, {{0, 1, 2}, {1, 0, 2}}, "perm_x1");
    CheckPerm(perm_x2_value, {{0, 1, 2}, {0, 2, 1}}, "perm_x2");
    CheckPerm(perm_y_value, {{1, 0, 2}}, "perm_y");

    TORCH_CHECK(x1.size(perm_x1_value[0]) == x2.size(perm_x2_value[0]), "x1 and x2 batch dimensions must match");
    TORCH_CHECK(x1.size(perm_x1_value[2]) == x2.size(perm_x2_value[1]), "x1 and x2 K dimensions must match");
    if (has_scale) {
        TORCH_CHECK(batch_split_factor_value == kDefaultBatchSplitFactor,
                    "batch_split_factor must be 1 when scale is provided");
        const int64_t scale_len = x2.size(perm_x2_value[0]) * x2.size(perm_x2_value[2]);
        TORCH_CHECK(scale->dim() == 1 && scale->size(0) == scale_len,
                    "scale must be 1-dimensional with length batch * N");
    }

    const std::vector<int64_t> output_shape = InferOutputShape(x1, x2, perm_x1_value, perm_x2_value, has_scale,
                                                               batch_split_factor_value);
    const at::ScalarType output_dtype = has_scale ? at::kChar : x1.scalar_type();
    at::Tensor out = at::empty(output_shape, x1.options().dtype(output_dtype));

    const at::IntArrayRef perm_x1_ref(perm_x1_value);
    const at::IntArrayRef perm_x2_ref(perm_x2_value);
    const at::IntArrayRef perm_y_ref(perm_y_value);
    const int8_t cube_math_type = enable_hf32 ? kUseHf32 : kKeepDtype;
    const int32_t batch_split_factor_i32 = static_cast<int32_t>(batch_split_factor_value);

    const bool is_weight_nz = at_npu::native::get_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ;
    if (is_weight_nz) {
        TORCH_CHECK(x1.scalar_type() == at::kHalf || x1.scalar_type() == at::kBFloat16,
                    "FRACTAL_NZ (WeightNZ) only supports float16/bfloat16 input, but got ", x1.scalar_type());
        ACLNN_CMD(aclnnTransposeBatchMatMulWeightNz, x1, x2, bias, scale, perm_x1_ref, perm_x2_ref, perm_y_ref,
                  cube_math_type, batch_split_factor_i32, out);
    } else {
        ACLNN_CMD(aclnnTransposeBatchMatMul, x1, x2, bias, scale, perm_x1_ref, perm_x2_ref, perm_y_ref, cube_math_type,
                  batch_split_factor_i32, out);
    }
    return out;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transpose_batch_mat_mul", &cann_ops_nn::matmul::transpose_batch_mat_mul, "TransposeBatchMatMul on NPU");
}
