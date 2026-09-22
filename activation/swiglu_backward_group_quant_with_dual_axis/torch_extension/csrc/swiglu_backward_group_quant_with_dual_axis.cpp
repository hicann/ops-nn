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

namespace cann_ops_nn::activation {
namespace {
constexpr int64_t FP8_E5M2 = 35;
constexpr int64_t FP8_E4M3FN = 36;

c10::SymInt Product(c10::SymIntArrayRef sizes, int64_t end)
{
    c10::SymInt result = 1;
    for (int64_t i = 0; i < end; ++i) {
        result *= sizes[i];
    }
    return result;
}

} // namespace

std::vector<at::Tensor> swiglu_backward_group_quant_with_dual_axis(const at::Tensor& gradY, const at::Tensor& x,
                                                                   const c10::optional<at::Tensor>& weight,
                                                                   const c10::optional<at::Tensor>& yOrigin,
                                                                   const c10::optional<at::Tensor>& groupIndex,
                                                                   double clampLimit, double alpha, double bias,
                                                                   int64_t quantMode, int64_t dstType)
{
    // Host tiling validates dtype, shape and attribute ranges. Keep only the rank
    // guard required to construct the -2-axis output shape below.
    TORCH_CHECK(x.dim() == 2, "x rank is invalid: actual=", x.dim(), ", expected = 2");

    const c10::SymInt totalRows = Product(x.sym_sizes(), x.dim() - 1);
    const bool hasGroup = groupIndex.has_value() && groupIndex->defined();
    const c10::SymInt groups = hasGroup ? groupIndex->sym_numel() : 1;
    const bool hasWeight = weight.has_value() && weight->defined();

    const bool useE5M2 = dstType == FP8_E5M2;
    const auto fp8Type = useE5M2 ? at::ScalarType::Float8_e5m2 : at::ScalarType::Float8_e4m3fn;
    const auto fp8AclType = useE5M2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
    auto fp8Options = x.options().dtype(fp8Type);
    auto scaleOptions = x.options().dtype(at::ScalarType::Float8_e8m0fnu);
    const c10::SymInt width = x.sym_size(-1);

    at::Tensor y1 = at::empty_symint(x.sym_sizes(), fp8Options);
    std::vector<c10::SymInt> scale1Shape(x.sym_sizes().begin(), x.sym_sizes().end() - 1);
    scale1Shape.push_back((width + 63) / 64);
    scale1Shape.push_back(2);
    at::Tensor scale1 = at::empty_symint(scale1Shape, scaleOptions);
    at::Tensor y2 = at::empty_symint(x.sym_sizes(), fp8Options);
    // NPU zero_ does not support E8M0; zero the byte storage, then reinterpret its bits.
    std::vector<c10::SymInt> scale2Shape;
    if (hasGroup) {
        scale2Shape = {totalRows / 64 + groups, width, 2};
    } else {
        scale2Shape.assign(x.sym_sizes().begin(), x.sym_sizes().end() - 2);
        scale2Shape.push_back((x.sym_size(-2) + 63) / 64);
        scale2Shape.push_back(width);
        scale2Shape.push_back(2);
    }
    at::Tensor scale2 = at::zeros_symint(scale2Shape, x.options().dtype(at::kByte))
                            .view(at::ScalarType::Float8_e8m0fnu);

    c10::optional<at::Tensor> gradWeightOut = c10::nullopt;
    at::Tensor gradWeight;
    if (hasWeight) {
        gradWeight = at::empty_symint(weight->sym_sizes(), weight->options());
        gradWeightOut = gradWeight;
    }

    TensorWrapper y1W = {y1, fp8AclType};
    TensorWrapper scale1W = {scale1, ACL_FLOAT8_E8M0};
    TensorWrapper y2W = {y2, fp8AclType};
    TensorWrapper scale2W = {scale2, ACL_FLOAT8_E8M0};
    ACLNN_CMD(aclnnSwigluBackwardGroupQuantWithDualAxis, gradY, x, weight, yOrigin, groupIndex, clampLimit, alpha, bias,
              quantMode, dstType, y1W, scale1W, y2W, scale2W, gradWeightOut);

    std::vector<at::Tensor> outputs = {y1, scale1, y2, scale2};
    if (hasWeight) {
        outputs.emplace_back(gradWeight);
    }
    return outputs;
}
} // namespace cann_ops_nn::activation

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_backward_group_quant_with_dual_axis",
          &cann_ops_nn::activation::swiglu_backward_group_quant_with_dual_axis,
          "swiglu_backward_group_quant_with_dual_axis on NPU");
}
