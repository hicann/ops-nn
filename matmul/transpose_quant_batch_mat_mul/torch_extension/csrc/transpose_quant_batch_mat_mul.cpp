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

namespace {
constexpr int64_t OFFSET_32_BITS = 32;
constexpr int64_t OFFSET_16_BITS = 16;
constexpr uint64_t GROUP_MAX = 65535UL;
constexpr size_t GROUP_DIM = 3;

int64_t check_and_get_group_size(at::IntArrayRef group_size_list)
{
    int64_t groups = 0;
    if (group_size_list.empty()) {
        return groups;
    }
    size_t group_dim = group_size_list.size();
    TORCH_CHECK(group_dim == GROUP_DIM, "group_sizes only support input with three elements, but got ", group_dim);
    int64_t group_m = static_cast<int64_t>(group_size_list[0]);
    int64_t group_n = static_cast<int64_t>(group_size_list[1]);
    int64_t group_k = static_cast<int64_t>(group_size_list[2]);
    bool invalid_group_param = ((group_m <= GROUP_MAX && group_m >= 0) && (group_n <= GROUP_MAX && group_n >= 0) &&
                                (group_k <= GROUP_MAX && group_k >= 0));
    TORCH_CHECK(invalid_group_param, "group param value must conform to range [0, 65535]");
    groups = static_cast<int64_t>((static_cast<uint64_t>(group_m) << OFFSET_32_BITS) +
                                  (static_cast<uint64_t>(group_n) << OFFSET_16_BITS) + static_cast<uint64_t>(group_k));
    return groups;
}
} // namespace

namespace cann_ops_nn {
namespace matmul {

at::Tensor transpose_quant_batch_mat_mul(
    const at::Tensor& x1, const at::Tensor& x2, int64_t dtype, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& x1_scale, const c10::optional<at::Tensor>& x2_scale,
    c10::optional<std::vector<int64_t>>& group_sizes, c10::optional<std::vector<int64_t>>& perm_x1,
    c10::optional<std::vector<int64_t>>& perm_x2, c10::optional<std::vector<int64_t>>& perm_y,
    c10::optional<int64_t> batch_split_factor, c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype,
    c10::optional<int64_t> x1_scale_dtype, c10::optional<int64_t> x2_scale_dtype)
{
    const at::Tensor& bias_real = bias.value_or(at::Tensor());
    const at::Tensor& x1_scale_real = x1_scale.value_or(at::Tensor());
    const at::Tensor& x2_scale_real = x2_scale.value_or(at::Tensor());

    // The torch interface only exposes the MX path: MXFP8 (float8_e4m3fn + e8m0
    // scale) and MXFP4 (float4_e2m1 + e8m0 scale).  Reject FP8-INT8 / Hifp8
    // inputs up-front so the user sees a clear error instead of a downstream
    // aclnn/graph failure.
    const aclDataType x1_acl = x1_dtype.has_value() ? GetAclDataType(x1_dtype.value()) :
                                                      ConvertToAclDataType(x1.scalar_type());
    const aclDataType x2_acl = x2_dtype.has_value() ? GetAclDataType(x2_dtype.value()) :
                                                      ConvertToAclDataType(x2.scalar_type());
    const bool x1_mx = (x1_acl == ACL_FLOAT4_E2M1) || (x1_acl == ACL_FLOAT8_E4M3FN);
    const bool x2_mx = (x2_acl == ACL_FLOAT4_E2M1) || (x2_acl == ACL_FLOAT8_E4M3FN);
    TORCH_CHECK(x1_mx, "x1 must be float4_e2m1 or float8_e4m3fn (MX fp4/fp8) in the torch interface, got aclDataType ",
                x1_acl);
    TORCH_CHECK(x2_mx, "x2 must be float4_e2m1 or float8_e4m3fn (MX fp4/fp8) in the torch interface, got aclDataType ",
                x2_acl);
    const aclDataType x1_scale_acl = x1_scale_dtype.has_value() ?
                                         GetAclDataType(x1_scale_dtype.value()) :
                                         (x1_scale_real.defined() ? ConvertToAclDataType(x1_scale_real.scalar_type()) :
                                                                    ACL_DT_UNDEFINED);
    const aclDataType x2_scale_acl = x2_scale_dtype.has_value() ?
                                         GetAclDataType(x2_scale_dtype.value()) :
                                         (x2_scale_real.defined() ? ConvertToAclDataType(x2_scale_real.scalar_type()) :
                                                                    ACL_DT_UNDEFINED);
    TORCH_CHECK(x1_scale_acl == ACL_DT_UNDEFINED || x1_scale_acl == ACL_FLOAT8_E8M0,
                "x1_scale must be float8_e8m0 (MX scale) in the torch interface, got aclDataType ", x1_scale_acl);
    TORCH_CHECK(x2_scale_acl == ACL_DT_UNDEFINED || x2_scale_acl == ACL_FLOAT8_E8M0,
                "x2_scale must be float8_e8m0 (MX scale) in the torch interface, got aclDataType ", x2_scale_acl);

    const int64_t b_idx = 0;
    const int64_t m_idx = 1;
    const int64_t ka_idx = 2;
    const int64_t kb_idx = 1;
    const int64_t n_idx = 2;
    const std::vector<int64_t> default_perm_x1 = {m_idx, b_idx, ka_idx};
    const std::vector<int64_t> default_perm_x2 = {b_idx, kb_idx, n_idx};
    const std::vector<int64_t> default_perm_y = {m_idx, b_idx, n_idx};
    const std::vector<int64_t> default_group_sizes = {0, 0, 0};

    const auto perm_x1_real = perm_x1.has_value() ? at::IntArrayRef(perm_x1.value()) : at::IntArrayRef(default_perm_x1);
    const auto perm_x2_real = perm_x2.has_value() ? at::IntArrayRef(perm_x2.value()) : at::IntArrayRef(default_perm_x2);
    const auto perm_y_real = perm_y.has_value() ? at::IntArrayRef(perm_y.value()) : at::IntArrayRef(default_perm_y);
    int64_t group_size_value = check_and_get_group_size(group_sizes.has_value() ? at::IntArrayRef(group_sizes.value()) :
                                                                                  at::IntArrayRef(default_group_sizes));
    int32_t batch_split_factor_value = static_cast<int32_t>(batch_split_factor.value_or(1));

    const bool x1_is_fp4 = Is4BitDtype(x1_acl);
    const bool x2_is_fp4 = Is4BitDtype(x2_acl);
    const int64_t x1_last_dim = static_cast<int64_t>(x1.sizes().size()) - 1;
    const int64_t x2_last_dim = static_cast<int64_t>(x2.sizes().size()) - 1;

    auto m_dim = x1.size(perm_x1_real[1]);
    if (x1_is_fp4 && perm_x1_real[1] == x1_last_dim) {
        m_dim *= FP4_IN_INT8;
    }
    auto batch_dim = x1.size(perm_x1_real[0]);
    auto n_dim = x2.size(perm_x2_real[2]);
    if (x2_is_fp4 && perm_x2_real[2] == x2_last_dim) {
        n_dim *= FP4_IN_INT8;
    }

    c10::SmallVector<int64_t, op_infer::SIZE> output_size = {m_dim, batch_dim, n_dim};

    if (batch_split_factor_value > 1) {
        output_size = {batch_split_factor_value, m_dim, batch_dim * n_dim / batch_split_factor_value};
    }

    aclDataType dtype_value = static_cast<aclDataType>(dtype);
    at::ScalarType scalar_type;
    if (dtype_value == ACL_FLOAT16) {
        scalar_type = at::ScalarType::Half;
    } else if (dtype_value == ACL_BF16) {
        scalar_type = at::ScalarType::BFloat16;
    } else {
        TORCH_CHECK(false, "unsupported output dtype, only support float16(1) and bfloat16(27), got ", dtype);
    }

    at::Tensor result = at::empty(output_size, at::TensorOptions().dtype(scalar_type).device(at::kPrivateUse1));

    bool is_nz = at_npu::native::get_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ;
    bool is_nd_nz_format = is_nz && at_npu::native::get_npu_format(x1) != ACL_FORMAT_FRACTAL_NZ;

    TensorWrapper x1_wrapper = {
        x1, x1_dtype.has_value() ? GetAclDataType(x1_dtype.value()) : ConvertToAclDataType(x1.scalar_type())};
    TensorWrapper x2_wrapper = {
        x2, x2_dtype.has_value() ? GetAclDataType(x2_dtype.value()) : ConvertToAclDataType(x2.scalar_type())};
    TensorWrapper result_wrapper = {result, dtype_value};

    aclDataType x1_scale_acltype = ACL_DT_UNDEFINED;
    if (x1_scale_dtype.has_value()) {
        x1_scale_acltype = GetAclDataType(x1_scale_dtype.value());
    } else if (x1_scale_real.defined()) {
        x1_scale_acltype = ConvertToAclDataType(x1_scale_real.scalar_type());
    }
    TensorWrapper x1_scale_wrapper = {x1_scale_real, x1_scale_acltype};

    aclDataType x2_scale_acltype = ACL_DT_UNDEFINED;
    if (x2_scale_dtype.has_value()) {
        x2_scale_acltype = GetAclDataType(x2_scale_dtype.value());
    } else if (x2_scale_real.defined()) {
        x2_scale_acltype = ConvertToAclDataType(x2_scale_real.scalar_type());
    }
    TensorWrapper x2_scale_wrapper = {x2_scale_real, x2_scale_acltype};
    int32_t dtype_value_i32 = static_cast<int32_t>(dtype_value);

    if (is_nd_nz_format) {
        auto weight_nz_func = GetOpApiFuncAddr("aclnnTransposeQuantBatchMatMulWeightNzGetWorkspaceSize");
        TORCH_CHECK(weight_nz_func != nullptr,
                    "In the current CANN version, aclnnTransposeQuantBatchMatMulWeightNz does not support "
                    "x2 as WeightNz input. Please upgrade the CANN package to version 9.1 or higher, "
                    "or set the x2 to ND mode.");
        ACLNN_CMD(aclnnTransposeQuantBatchMatMulWeightNz, x1_wrapper, x2_wrapper, bias_real, x1_scale_wrapper,
                  x2_scale_wrapper, dtype_value_i32, group_size_value, perm_x1_real, perm_x2_real, perm_y_real,
                  batch_split_factor_value, result_wrapper);
    } else {
        ACLNN_CMD(aclnnTransposeQuantBatchMatMul, x1_wrapper, x2_wrapper, bias_real, x1_scale_wrapper, x2_scale_wrapper,
                  dtype_value_i32, group_size_value, perm_x1_real, perm_x2_real, perm_y_real, batch_split_factor_value,
                  result_wrapper);
    }

    return result;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transpose_quant_batch_mat_mul", &cann_ops_nn::matmul::transpose_quant_batch_mat_mul,
          "TransposeQuantBatchMatMul on NPU");
}
