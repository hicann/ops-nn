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
namespace activation {
namespace {
constexpr int64_t kSplitFactor = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckContiguous(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
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

void CheckXShape(const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    int64_t normalizedDim = dim < 0 ? dim + x.dim() : dim;
    TORCH_CHECK(normalizedDim >= 0 && normalizedDim < x.dim(), "dim should be in range [", -x.dim(), ", ", x.dim() - 1,
                "], but got ", dim);
    TORCH_CHECK(x.size(normalizedDim) % kSplitFactor == 0, "x[dim] must be divisible by 2, but got ",
                x.size(normalizedDim));
}

void CheckGradShape(const at::Tensor& yGrad, const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(yGrad.dim() == x.dim(), "grad_y rank (", yGrad.dim(), ") must match x rank (", x.dim(), ")");
    int64_t normalizedDim = dim < 0 ? dim + x.dim() : dim;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i == normalizedDim) {
            TORCH_CHECK(yGrad.size(i) == x.size(i) / kSplitFactor, "grad_y dim ", i, " (", yGrad.size(i),
                        ") must be half of x dim ", i, " (", x.size(i), ")");
        } else {
            TORCH_CHECK(yGrad.size(i) == x.size(i), "grad_y dim ", i, " (", yGrad.size(i), ") must match x dim ", i,
                        " (", x.size(i), ")");
        }
    }
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
} // namespace

at::Tensor clipped_swiglu_grad(const at::Tensor& grad_y, const at::Tensor& x,
                               const c10::optional<at::Tensor>& group_index, int64_t dim, double alpha, double limit,
                               double bias, bool interleaved, int64_t clamp_mode)
{
    CheckNpuTensor(grad_y, "grad_y");
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(group_index, "group_index");

    CheckContiguous(grad_y, "grad_y");
    CheckContiguous(x, "x");
    CheckOptionalContiguous(group_index, "group_index");

    TORCH_CHECK(IsSupportedDtype(x.scalar_type()), "x dtype must be float16, bfloat16 or float32, but got ",
                x.scalar_type());
    TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(), "grad_y dtype (", grad_y.scalar_type(),
                ") must match x dtype (", x.scalar_type(), ")");

    CheckXShape(x, dim);
    CheckGradShape(grad_y, x, dim);
    CheckOptionalGroupIndex(group_index);

    TORCH_CHECK(limit > 0, "limit must be positive, but got ", limit);
    TORCH_CHECK(clamp_mode == 0, "clampMode only supports 0, but got ", clamp_mode);

    at::Tensor grad_x = at::empty(x.sizes(), x.options());

    ACLNN_CMD(aclnnClippedSwigluGrad, grad_y, x, group_index, dim, alpha, limit, bias, interleaved, grad_x);
    return grad_x;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("clipped_swiglu_grad", &cann_ops_nn::activation::clipped_swiglu_grad, "ClippedSwigluGrad on NPU");
}
