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
#include "aclnnop/aclnn_flat_quant_v3.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

constexpr int64_t QUINT4X2_DTYPE = 16;
constexpr int64_t FLOAT4_E2M1FN_X2_DTYPE = 296;

std::tuple<at::Tensor, at::Tensor> flat_quant(const at::Tensor& x, const at::Tensor& kronecker_p1,
                                              const at::Tensor& kronecker_p2, double clip_ratio, int64_t dst_dtype,
                                              double dst_type_max, const c10::optional<at::Tensor>& group_list,
                                              int64_t group_list_type)
{
    // 1. 设备检查
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    TORCH_CHECK(kronecker_p1.device().type() == at::kPrivateUse1, "kronecker_p1 must be on NPU device");
    TORCH_CHECK(kronecker_p2.device().type() == at::kPrivateUse1, "kronecker_p2 must be on NPU device");

    // 2. 维度检查
    TORCH_CHECK(x.dim() == 3, "x must be 3-dimensional [M, N1, N2], but got ", x.dim());
    int64_t m = x.size(0);
    int64_t n1 = x.size(1);
    int64_t n2 = x.size(2);

    TORCH_CHECK(kronecker_p1.dim() == 2, "kronecker_p1 must be 2-dimensional [N1, N1], but got ", kronecker_p1.dim());
    TORCH_CHECK(kronecker_p1.size(0) == n1 && kronecker_p1.size(1) == n1, "kronecker_p1 shape must be [", n1, ", ", n1,
                "], but got [", kronecker_p1.size(0), ", ", kronecker_p1.size(1), "]");

    TORCH_CHECK(kronecker_p2.dim() == 2, "kronecker_p2 must be 2-dimensional [N2, N2], but got ", kronecker_p2.dim());
    TORCH_CHECK(kronecker_p2.size(0) == n2 && kronecker_p2.size(1) == n2, "kronecker_p2 shape must be [", n2, ", ", n2,
                "], but got [", kronecker_p2.size(0), ", ", kronecker_p2.size(1), "]");

    // 3. 数据类型检查
    auto input_dtype = x.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(kronecker_p1.scalar_type() == input_dtype, "kronecker_p1 dtype must match x dtype: ", input_dtype,
                " vs ", kronecker_p1.scalar_type());
    TORCH_CHECK(kronecker_p2.scalar_type() == input_dtype, "kronecker_p2 dtype must match x dtype: ", input_dtype,
                " vs ", kronecker_p2.scalar_type());

    // 4. 可选参数检查
    if (group_list.has_value() && group_list.value().defined()) {
        TORCH_CHECK(group_list.value().device().type() == at::kPrivateUse1, "group_list must be on NPU device");
        TORCH_CHECK(group_list.value().scalar_type() == at::kLong, "group_list dtype must be INT64, but got ",
                    group_list.value().scalar_type());
        auto group_shape = group_list.value().sizes();
        TORCH_CHECK(group_shape.size() == 1 || group_shape.size() == 2, "group_list must be 1D or 2D, but got ",
                    group_shape.size(), "D");
    }

    // 5. 参数范围检查
    TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0, "clip_ratio must be in range (0, 1], but got ", clip_ratio);
    TORCH_CHECK(dst_type_max == 0.0 || (dst_type_max >= 6.0 && dst_type_max <= 12.0),
                "dst_type_max must be 0 or in range [6, 12], but got ", dst_type_max);
    TORCH_CHECK(group_list_type >= 0 && group_list_type <= 2, "group_list_type must be in range [0, 2], but got ",
                group_list_type);

    // 6. 确定输出数据类型和形状
    aclDataType out_acl_type = ACL_INT32;
    aclDataType scale_acl_type = ACL_FLOAT;
    if (dst_dtype == QUINT4X2_DTYPE) {
        // 如果dtype为torch.quint4x2时, 输出tensor类型为INT32, 由8个INT4拼接。
        TORCH_CHECK(n2 % 8 == 0, "N2 must be divisible by 8 for INT4 output, but got N2=", n2);
    } else if (dst_dtype == FLOAT4_E2M1FN_X2_DTYPE) {
        out_acl_type = ACL_FLOAT4_E2M1;
        scale_acl_type = ACL_FLOAT8_E8M0;
        TORCH_CHECK(n2 % 2 == 0, "N2 must be even for FP4 output, but got N2=", n2);
    } else {
        TORCH_CHECK(false, "dst_dtype must be INT4 or FP4, got dst_dtype=", dst_dtype);
    }

    // 7. 创建输出张量
    std::vector<int64_t> out_shape;
    std::vector<int64_t> quant_scale_shape;
    if (out_acl_type == ACL_INT32) {
        // INT4: shape为 [M, N1, N2/8]
        out_shape = {m, n1, n2 / 8};
        quant_scale_shape = {m};
    } else {
        // FP4: shape与x一致 [M, N1 * N2]
        out_shape = {m, n1 * n2 / 2};
        quant_scale_shape = {m, (n1 * n2 + 63) / 64, 2};
    }

    at::ScalarType out_scalar_type = out_acl_type == ACL_INT32 ? at::kInt : at::kByte;
    at::Tensor out = at::empty(out_shape, at::TensorOptions().dtype(out_scalar_type).device(at::kPrivateUse1));
    at::ScalarType scale_scalar_type = out_acl_type == ACL_INT32 ? at::kFloat : at::kByte;
    at::Tensor quant_scale = at::empty(quant_scale_shape,
                                       at::TensorOptions().dtype(scale_scalar_type).device(at::kPrivateUse1));

    // 8. ACLNN 调用
    TensorWrapper out_wrapper{out, out_acl_type};
    TensorWrapper scale_wrapper{quant_scale, scale_acl_type};
    ACLNN_CMD(aclnnFlatQuantV3, x, kronecker_p1, kronecker_p2, group_list, clip_ratio, dst_type_max, group_list_type,
              out_wrapper, scale_wrapper);

    return std::make_tuple(out, quant_scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flat_quant", &cann_ops_nn::quant::flat_quant, "FlatQuant quantization on NPU");
}
