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
namespace {

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

void CheckGradInputShape(const at::Tensor& grad_output, const at::Tensor& x)
{
    TORCH_CHECK(grad_output.dim() == 2 || grad_output.dim() == 3, "grad_output must be 2D or 3D, but got ",
                grad_output.dim(), "D");
    TORCH_CHECK(x.dim() == grad_output.dim(), "x rank must equal grad_output rank");
    for (int64_t dim = 0; dim < grad_output.dim() - 1; ++dim) {
        TORCH_CHECK(x.size(dim) == grad_output.size(dim), "x.shape[", dim, "] must equal grad_output.shape[", dim, "]");
    }
    TORCH_CHECK(x.size(-1) == grad_output.size(-1) * 2, "x.shape[-1] must equal 2 * grad_output.shape[-1]");
    TORCH_CHECK(grad_output.size(-1) > 0, "grad_output.shape[-1] must be greater than 0");
}

void CheckGradInputDtype(const at::Tensor& grad_output, const at::Tensor& x)
{
    TORCH_CHECK(grad_output.scalar_type() == at::kHalf || grad_output.scalar_type() == at::kBFloat16 ||
                    grad_output.scalar_type() == at::kFloat,
                "grad_output must have FLOAT16, BFLOAT16, or FLOAT dtype");
    TORCH_CHECK(x.scalar_type() == grad_output.scalar_type(), "x dtype must equal grad_output dtype");
}

} // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>> swiglu_group_backward(
    const at::Tensor& grad_output, const at::Tensor& x, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& y_origin, const c10::optional<at::Tensor>& group_index, double clamp_limit)
{
    CheckNpuTensor(grad_output, "grad_output");
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(y_origin, "y_origin");
    CheckOptionalNpuTensor(group_index, "group_index");
    CheckGradInputShape(grad_output, x);
    CheckGradInputDtype(grad_output, x);
    TORCH_CHECK(clamp_limit >= 0.0, "clamp_limit must be >= 0.0");

    const bool has_weight = weight.has_value() && weight.value().defined();
    const bool has_y_origin = y_origin.has_value() && y_origin.value().defined();
    TORCH_CHECK(has_weight == has_y_origin, "weight and y_origin must be provided together");

    at::Tensor grad_x = at::empty(x.sizes(), grad_output.options());
    at::Tensor grad_weight = at::empty({0}, grad_output.options().dtype(at::kFloat));

    if (has_weight) {
        int64_t weightElementNum = weight.value().numel();
        int64_t totalRows = 1;
        for (int64_t i = 0; i < grad_output.dim() - 1; ++i) {
            totalRows *= grad_output.size(i);
        }
        TORCH_CHECK(weightElementNum == totalRows, "weight element num must equal total rows (", totalRows,
                    "), but got ", weightElementNum);
        TORCH_CHECK(weight.value().scalar_type() == at::kFloat, "weight dtype must be FLOAT");
        grad_weight = at::empty(weight.value().sizes(), weight.value().options().dtype(at::kFloat));
    }

    if (has_y_origin) {
        TORCH_CHECK(y_origin.value().sizes() == grad_output.sizes(), "y_origin shape must equal grad_output shape");
        TORCH_CHECK(y_origin.value().scalar_type() == grad_output.scalar_type(),
                    "y_origin dtype must equal grad_output dtype");
    }
    if (group_index.has_value() && group_index.value().defined()) {
        TORCH_CHECK(group_index.value().dim() == 1, "group_index must be 1D when present");
        TORCH_CHECK(group_index.value().numel() > 0, "group_index must not be empty when present");
        TORCH_CHECK(group_index.value().scalar_type() == at::kLong, "group_index dtype must be INT64");
    }

    float clamp_limit_float = static_cast<float>(clamp_limit);
    ACLNN_CMD(aclnnSwigluGroupGrad, grad_output, x, weight, y_origin, group_index, clamp_limit_float, grad_x,
              grad_weight);
    if (has_weight) {
        return std::make_tuple(grad_x, c10::optional<at::Tensor>(grad_weight));
    }
    return std::make_tuple(grad_x, c10::nullopt);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group_backward", &cann_ops_nn::quant::swiglu_group_backward, "SwigluGroupBackward on NPU");
}
