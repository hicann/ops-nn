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
namespace activation {
namespace {
constexpr int64_t kSplitFactor = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
}

void CheckContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalContiguous(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckContiguous(tensor.value(), name);
    }
}

bool IsSupportedDtype(const at::ScalarType& dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

void CheckXShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    const int64_t lastDim = x.size(x.dim() - 1);
    TORCH_CHECK(lastDim % kSplitFactor == 0, "x last dim size should be even");
}

void CheckGradYShape(const at::Tensor& gradY, const at::Tensor& x)
{
    TORCH_CHECK(gradY.dim() == x.dim(), "gradY rank (", gradY.dim(), ") must match x rank (", x.dim(), ")");
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        TORCH_CHECK(gradY.size(i) == x.size(i), "gradY dim ", i, " (", gradY.size(i), ") must match x dim ", i, " (",
                    x.size(i), ")");
    }
    const int64_t xLastDim = x.size(x.dim() - 1);
    TORCH_CHECK(gradY.size(gradY.dim() - 1) == xLastDim / kSplitFactor, "gradY last dim (", gradY.size(gradY.dim() - 1),
                ") must be half of x last dim (", xLastDim, ")");
}

void CheckOptionalWeight(const c10::optional<at::Tensor>& weight, const at::Tensor& gradY,
                         const c10::optional<at::Tensor>& yOrigin)
{
    if (!weight.has_value() || !weight.value().defined()) {
        return;
    }
    const auto& w = weight.value();
    TORCH_CHECK(w.scalar_type() == at::kFloat, "weight dtype must be float32, but got ", w.scalar_type());
    TORCH_CHECK(w.dim() == gradY.dim(), "weight rank (", w.dim(), ") must match gradY rank (", gradY.dim(), ")");
    for (int64_t i = 0; i < gradY.dim() - 1; ++i) {
        TORCH_CHECK(w.size(i) == gradY.size(i), "weight dim ", i, " (", w.size(i), ") must match gradY dim ", i, " (",
                    gradY.size(i), ")");
    }
    TORCH_CHECK(w.size(w.dim() - 1) == 1, "weight last dim must be 1, but got ", w.size(w.dim() - 1));
    TORCH_CHECK(yOrigin.has_value() && yOrigin.value().defined(), "y_origin must be provided when weight is provided");
}

void CheckOptionalYOrigin(const c10::optional<at::Tensor>& yOrigin, const at::Tensor& gradY)
{
    if (!yOrigin.has_value() || !yOrigin.value().defined()) {
        return;
    }
    const auto& yo = yOrigin.value();
    TORCH_CHECK(yo.scalar_type() == gradY.scalar_type(), "y_origin dtype (", yo.scalar_type(),
                ") must match gradY dtype (", gradY.scalar_type(), ")");
    TORCH_CHECK(yo.sizes().equals(gradY.sizes()), "y_origin shape ", yo.sizes(), " must match gradY shape ",
                gradY.sizes());
}

void CheckOptionalGroupIndex(const c10::optional<at::Tensor>& groupIndex)
{
    if (!groupIndex.has_value() || !groupIndex.value().defined()) {
        return;
    }
    const auto& gi = groupIndex.value();
    TORCH_CHECK(gi.scalar_type() == at::kLong, "group_index dtype must be int64, but got ", gi.scalar_type());
    TORCH_CHECK(gi.dim() == 1, "group_index must be 1-dimensional, but got rank ", gi.dim());
}

void CheckClampLimit(double clampLimit)
{
    TORCH_CHECK(clampLimit == -1.0 || clampLimit > 0.0, "clamp_limit must be -1.0 or > 0.0, but got ", clampLimit);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> swiglu_group_quant_backward(const at::Tensor& grad_y, const at::Tensor& x,
                                                               const c10::optional<at::Tensor>& weight,
                                                               const c10::optional<at::Tensor>& y_origin,
                                                               const c10::optional<at::Tensor>& group_index,
                                                               double clamp_limit)
{
    CheckNpuTensor(grad_y, "grad_y");
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(y_origin, "y_origin");
    CheckOptionalNpuTensor(group_index, "group_index");

    CheckContiguous(grad_y, "grad_y");
    CheckContiguous(x, "x");
    CheckOptionalContiguous(weight, "weight");
    CheckOptionalContiguous(y_origin, "y_origin");
    CheckOptionalContiguous(group_index, "group_index");

    TORCH_CHECK(IsSupportedDtype(x.scalar_type()), "x dtype must be float16, bfloat16 or float32, but got ",
                x.scalar_type());
    TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(), "grad_y dtype (", grad_y.scalar_type(),
                ") must match x dtype (", x.scalar_type(), ")");

    CheckXShape(x);
    CheckGradYShape(grad_y, x);
    CheckOptionalYOrigin(y_origin, grad_y);
    CheckOptionalWeight(weight, grad_y, y_origin);
    CheckOptionalGroupIndex(group_index);
    CheckClampLimit(clamp_limit);

    // grad_x: same shape and dtype as x
    at::Tensor grad_x = at::empty(x.sizes(), x.options());

    // grad_weight: same shape as weight if provided, otherwise empty
    at::Tensor grad_weight = at::empty({0}, x.options().dtype(at::kFloat));
    if (weight.has_value() && weight.value().defined()) {
        grad_weight = at::empty(weight.value().sizes(), x.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnSwigluGroupQuantGrad, grad_y, x, weight, y_origin, group_index, clamp_limit, grad_x, grad_weight);
    return std::make_tuple(grad_x, grad_weight);
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group_quant_backward", &cann_ops_nn::activation::swiglu_group_quant_backward,
          "SwigluGroupQuantGrad on NPU");
}
