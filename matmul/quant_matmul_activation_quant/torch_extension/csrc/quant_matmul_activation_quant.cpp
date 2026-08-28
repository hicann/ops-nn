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
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
static const size_t OFFSET_32_BITS = 32;
static const size_t OFFSET_16_BITS = 16;
static const uint64_t GROUP_MAX = 65535UL;
static const size_t GROUP_DIM = 3;

int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t value_num = 0;
    if (factor == 0) {
        return value_num;
    }
    if (value % factor == 0) {
        value_num = value / factor;
    } else {
        value_num = value / factor + 1;
    }

    return value_num;
}

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

bool static is_transpose_last_two_dims(const at::Tensor& tensor)
{
    if (tensor.dim() < 2 || tensor.dim() > 6) {
        return false;
    }
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        int64_t tmpNxD = tensor.size(dim1) * tensor.size(dim2);
        for (int64_t batchDim = tensor.dim() - 3; batchDim >= 0; batchDim--) {
            if (tensor.stride(batchDim) != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor.size(batchDim);
        }
        if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

} // namespace

namespace cann_ops_nn {
namespace matmul {

std::tuple<at::Tensor, at::Tensor> quant_matmul_activation_quant(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& x2_scale, const c10::optional<at::Tensor>& x1_scale,
    const c10::optional<at::Tensor>& bias, c10::optional<int64_t> output_dtype, c10::optional<int64_t> x1_dtype,
    c10::optional<int64_t> x2_dtype, c10::optional<int64_t> x1scale_dtype, c10::optional<int64_t> x2scale_dtype,
    c10::optional<std::vector<int64_t>>& group_sizes, c10::string_view activation_type, c10::string_view quant_mode,
    c10::string_view round_mode, c10::optional<int64_t> scale_alg, double dst_type_max)
{
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();

    int64_t x1_a = x1.size(x1_dim_num - 2);
    int64_t x1_b = x1.size(x1_dim_num - 1);
    int64_t x2_a = x2.size(x2_dim_num - 2);
    int64_t x2_b = x2.size(x2_dim_num - 1);

    int64_t m = 0;
    int64_t n = 0;
    bool transposeX1 = false;
    bool transposeX2 = false;

    constexpr int64_t DTYPE_FLOAT4_E2M1 = static_cast<int64_t>(DType::FLOAT4_E2M1);

    int64_t x1_dtype_val = x1_dtype.value_or(0);
    int64_t x2_dtype_val = x2_dtype.value_or(0);
    bool isMXFP4 = x1_dtype_val == DTYPE_FLOAT4_E2M1 && x2_dtype_val == DTYPE_FLOAT4_E2M1;

    if (isMXFP4) {
        bool transX1 = is_transpose_last_two_dims(x1);
        bool transX2 = is_transpose_last_two_dims(x2);
        m = !transX1 ? x1_a : x1_a * FP4_IN_INT8;
        n = transX2 ? x2_b : x2_b * FP4_IN_INT8;
    } else {
        m = x1_a;
        n = x2_b;
    }

    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    auto x1_batch_num = x1_dim_num - 2;
    auto x2_batch_num = x2_dim_num - 2;
    auto max_batch_num = std::max(x1_batch_num, x2_batch_num);
    for (int64_t i = 0; i < max_batch_num; i++) {
        int64_t x1_idx = x1_batch_num - (max_batch_num - i);
        int64_t x2_idx = x2_batch_num - (max_batch_num - i);
        int64_t x1_dim = (x1_idx >= 0) ? x1.size(x1_idx) : 1;
        int64_t x2_dim = (x2_idx >= 0) ? x2.size(x2_idx) : 1;
        TORCH_CHECK(x1_dim == x2_dim || x1_dim == 1 || x2_dim == 1,
                    "batch dims are not broadcastable: x1 batch dim = ", x1_dim, ", x2 batch dim = ", x2_dim);
        output_size.push_back(std::max(x1_dim, x2_dim));
    }
    output_size.push_back(m);
    output_size.push_back(n);

    c10::SmallVector<int64_t, op_infer::SIZE> scale_size = op_infer::array_to_small_vector(output_size);
    int64_t axis_change = output_size.size() - 1;
    int64_t dim_size = CeilDiv(scale_size[axis_change], BLOCK_SIZE);
    dim_size = (dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
    scale_size[axis_change] = dim_size;
    scale_size.emplace_back(ALIGN_NUM);

    aclDataType output_acltype;
    int64_t output_dtype_val = output_dtype.value_or(0);
    bool special_output_type = false;

    if (output_dtype_val == DTYPE_FLOAT4_E2M1) {
        special_output_type = true;
    }

    at::Tensor output;
    if (special_output_type) {
        int64_t last_dim_val = output_size[output_size.size() - 1];
        TORCH_CHECK(last_dim_val % 2 == 0, "The last dim output shape must be divisible by 2 if "
                                           "output dtype is FLOAT4_E2M1");
        output_size[output_size.size() - 1] = last_dim_val / 2;
        output = at::empty(output_size, at::TensorOptions().dtype(c10::ScalarType::Byte).device(at::kPrivateUse1));
        output_acltype = GetAclDataType(output_dtype_val);
    } else if (output_dtype.has_value()) {
        at::ScalarType scalar_dtype = at::ScalarType::Float8_e5m2;
        output_acltype = GetAclDataType(output_dtype_val);
        if (output_acltype == ACL_FLOAT8_E5M2) {
            scalar_dtype = at::ScalarType::Float8_e5m2;
        } else if (output_acltype == ACL_FLOAT8_E4M3FN) {
            scalar_dtype = at::ScalarType::Float8_e4m3fn;
        } else {
            TORCH_CHECK(false,
                        "unsupport output_dtype, only support output_dtype FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT4_E2M1");
        }
        output = at::empty(output_size, at::TensorOptions().dtype(scalar_dtype).device(at::kPrivateUse1));
    } else {
        output_acltype = ConvertToAclDataType(x1.scalar_type());
        output = at::empty(output_size, at::TensorOptions().dtype(x1.scalar_type()).device(at::kPrivateUse1));
    }

    at::Tensor scale_cpu = at::zeros(scale_size, at::TensorOptions().dtype(c10::ScalarType::Float8_e8m0fnu));
    at::Tensor scale = scale_cpu.to(at::TensorOptions().device(at::kPrivateUse1));

    TensorWrapper output_wrapper = {output, output_acltype};
    TensorWrapper scale_wrapper = {scale, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper x1_wrapper = {
        x1, x1_dtype.has_value() ? GetAclDataType(x1_dtype.value()) : ConvertToAclDataType(x1.scalar_type())};
    TensorWrapper x2_wrapper = {
        x2, x2_dtype.has_value() ? GetAclDataType(x2_dtype.value()) : ConvertToAclDataType(x2.scalar_type())};

    at::Tensor x1_scale_real = x1_scale.has_value() ? x1_scale.value() : at::Tensor();

    aclDataType x1_scale_acltype = ACL_DT_UNDEFINED;
    if (x1scale_dtype.has_value()) {
        x1_scale_acltype = GetAclDataType(x1scale_dtype.value());
    } else if (x1_scale_real.defined()) {
        x1_scale_acltype = ConvertToAclDataType(x1_scale_real.scalar_type());
    }
    TensorWrapper x1_scale_wrapper = {x1_scale_real, x1_scale_acltype};
    TensorWrapper x2_scale_wrapper = {x2_scale, x2scale_dtype.has_value() ?
                                                    GetAclDataType(x2scale_dtype.value()) :
                                                    ConvertToAclDataType(x2_scale.scalar_type())};

    int64_t scale_alg_optional = scale_alg.has_value() ? scale_alg.value() : DEFAULT_SCALE_ALG;

    at::IntArrayRef group_size_ref = group_sizes.has_value() ? at::IntArrayRef(group_sizes.value()) : at::IntArrayRef{};
    int64_t group_size = check_and_get_group_size(group_size_ref);
    TORCH_CHECK(group_size != -1, "Invalid group_sizes.");

    char* activation_type_ptr = const_cast<char*>(activation_type.data());
    char* quant_mode_ptr = const_cast<char*>(quant_mode.data());
    char* round_mode_ptr = const_cast<char*>(round_mode.data());

    bool isX2Nz = at_npu::native::get_npu_format(x2) == static_cast<int64_t>(ACL_FORMAT_FRACTAL_NZ);
    if (isX2Nz) {
        ACLNN_CMD(aclnnQuantMatmulActivationQuantWeightNz, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper,
                  bias, transposeX1, transposeX2, group_size, activation_type_ptr, quant_mode_ptr, round_mode_ptr,
                  scale_alg_optional, dst_type_max, output_wrapper, scale_wrapper);
    } else {
        ACLNN_CMD(aclnnQuantMatmulActivationQuant, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, bias,
                  transposeX1, transposeX2, group_size, activation_type_ptr, quant_mode_ptr, round_mode_ptr,
                  scale_alg_optional, dst_type_max, output_wrapper, scale_wrapper);
    }

    return std::make_tuple(output, scale);
}
} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_matmul_activation_quant", &cann_ops_nn::matmul::quant_matmul_activation_quant,
          "QuantMatmulActivationQuant on NPU");
}
