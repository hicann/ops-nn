/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <tuple>
#include <vector>
#include <torch/extension.h>
#include "aclnnop/aclnn_rotate_quant_v2.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {
namespace {
constexpr int64_t INT4_VALUES_PER_INT32 = 8;
constexpr int64_t FP4_VALUES_PER_BYTE = 2;
constexpr int64_t MX_SCALE_PAIR_SIZE = 64;
constexpr int64_t MX_SCALES_PER_PAIR = 2;
constexpr int64_t MIN_INPUT_RANK = 1;
constexpr int64_t MAX_INPUT_RANK = 7;
constexpr int64_t MATRIX_RANK = 2;
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 0;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;

void CheckInputDevice(const at::Tensor& input, const at::Tensor& x, const char* name)
{
    TORCH_CHECK(input.defined(), name, " must be defined");
    TORCH_CHECK(input.device() == x.device(), name, " must be on ", x.device(), ", but got ", input.device());
}
} // namespace

std::tuple<at::Tensor, at::Tensor> rotate_quant(const at::Tensor& x, const at::Tensor& rotation,
                                                const c10::optional<at::Tensor>& alpha,
                                                c10::optional<int64_t> dst_dtype, c10::optional<int64_t> axis,
                                                const c10::optional<std::string>& round_mode,
                                                c10::optional<int64_t> scale_alg, c10::optional<double> dst_type_max,
                                                c10::optional<bool> transpose_y,
                                                const c10::optional<at::Tensor>& group_list, int64_t group_list_type)
{
    TORCH_CHECK(x.defined() && x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    CheckInputDevice(rotation, x, "rotation");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(rotation.scalar_type() == x.scalar_type(), "x and rotation must have the same dtype, but got ",
                x.scalar_type(), " and ", rotation.scalar_type());
    TORCH_CHECK(x.dim() >= MIN_INPUT_RANK && x.dim() <= MAX_INPUT_RANK,
                "x must have between 1 and 7 dimensions, but got ", x.dim());
    TORCH_CHECK(group_list_type == GROUP_LIST_TYPE_CUMSUM || group_list_type == GROUP_LIST_TYPE_COUNT,
                "group_list_type must be 0 or 1, but got ", group_list_type);
    if (alpha.has_value()) {
        CheckInputDevice(alpha.value(), x, "alpha");
    }
    const bool grouped = group_list.has_value();
    if (grouped) {
        CheckInputDevice(group_list.value(), x, "group_list");
    }

    const int64_t dst_dtype_value = dst_dtype.value_or(static_cast<int64_t>(at::kChar));
    const bool is_int4 = dst_dtype_value == static_cast<int64_t>(at::ScalarType::QUInt4x2);
    const aclDataType out_acl_type = is_int4 ? ACL_INT32 : GetAclDataType(dst_dtype_value);
    at::ScalarType out_scalar_type;
    switch (out_acl_type) {
        case ACL_INT8:
            out_scalar_type = at::kChar;
            break;
        case ACL_INT32:
            TORCH_CHECK(is_int4, "INT4 output must be requested with dst_dtype=torch.quint4x2");
            out_scalar_type = at::kInt;
            break;
        case ACL_FLOAT4_E2M1:
            TORCH_CHECK(dst_dtype_value == static_cast<int64_t>(ACL_FLOAT4_E2M1) + g_toAclOffset,
                        "FP4 output must be requested with dst_dtype=torch_npu.float4_e2m1fn_x2");
            out_scalar_type = at::kByte;
            break;
        case ACL_FLOAT8_E5M2:
            out_scalar_type = at::kFloat8_e5m2;
            break;
        case ACL_FLOAT8_E4M3FN:
            out_scalar_type = at::kFloat8_e4m3fn;
            break;
        default:
            TORCH_CHECK(false, "unsupported dst_dtype for rotate_quant: ", dst_dtype_value);
    }
    const bool is_mx = out_acl_type == ACL_FLOAT4_E2M1 || out_acl_type == ACL_FLOAT8_E5M2 ||
                       out_acl_type == ACL_FLOAT8_E4M3FN;
    TORCH_CHECK(!grouped || is_mx, "grouped rotate_quant only supports MXFP4/MXFP8 output");
    TORCH_CHECK(is_mx || x.dim() == MATRIX_RANK, "INT8/INT4 x must be 2-dimensional [M,N]");
    TORCH_CHECK(!grouped || x.dim() == MATRIX_RANK, "grouped x must be 2-dimensional [M,N]");

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    const int64_t n = out_shape.back();
    if (is_int4) {
        TORCH_CHECK(n % INT4_VALUES_PER_INT32 == 0, "INT4 N must be divisible by 8");
        out_shape.back() /= INT4_VALUES_PER_INT32;
    } else if (out_acl_type == ACL_FLOAT4_E2M1) {
        TORCH_CHECK(n % FP4_VALUES_PER_BYTE == 0, "FP4 N must be divisible by 2");
        out_shape.back() /= FP4_VALUES_PER_BYTE;
    }
    std::vector<int64_t> scale_shape;
    if (is_mx) {
        scale_shape.assign(x.sizes().begin(), x.sizes().end());
        scale_shape.back() = n / MX_SCALE_PAIR_SIZE + (n % MX_SCALE_PAIR_SIZE != 0);
        scale_shape.push_back(MX_SCALES_PER_PAIR);
    } else {
        scale_shape = {x.size(0)};
    }
    auto y = at::empty(out_shape, x.options().dtype(out_scalar_type));
    auto scale = at::empty(scale_shape, x.options().dtype(is_mx ? at::kByte : at::kFloat));
    // Keep FP4 output packed in uint8 for Torch versions without native FP4 support.
    // Expose E8M0 scales with their native dtype for subsequent operators.
    if (is_mx) {
        scale = scale.view(at::ScalarType::Float8_e8m0fnu);
    }
    // Torch counts packed FP4 bytes; ACLNN uses logical FP4 elements.
    TensorWrapper y_wrapper{y, out_acl_type, out_acl_type == ACL_FLOAT4_E2M1 ? y.dim() - 1 : -1};
    TensorWrapper scale_wrapper{scale, is_mx ? ACL_FLOAT8_E8M0 : ACL_FLOAT};

    const int64_t axis_value = axis.value_or(-1);
    const int64_t scale_alg_value = scale_alg.value_or(0);
    const double dst_type_max_value = dst_type_max.value_or(0.0);
    const bool transpose_y_value = transpose_y.value_or(false);
    std::string round_mode_value = round_mode.value_or("rint");
    char* round_mode_ptr = round_mode_value.data();
    // Preserve input strides: ACLNN performs the same Contiguous conversion as ordinary RotateQuant.
    ACLNN_CMD(aclnnRotateQuantV2, x, rotation, alpha, group_list, group_list_type, axis_value, round_mode_ptr,
              scale_alg_value, dst_type_max_value, transpose_y_value, y_wrapper, scale_wrapper);
    return std::make_tuple(y, scale);
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rotate_quant", &cann_ops_nn::matmul::rotate_quant, "RotateQuant on NPU");
}
