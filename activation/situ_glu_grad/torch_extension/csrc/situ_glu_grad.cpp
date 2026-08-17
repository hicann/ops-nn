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

void ValidateDim(const at::Tensor& x, int64_t dim)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    int64_t actualDim = dim;
    if (actualDim < 0) {
        actualDim += x.dim();
    }
    TORCH_CHECK(actualDim >= 0 && actualDim < x.dim(), "dim should be in [", -x.dim(), ", ", x.dim() - 1, "], but got ",
                dim);
    const int64_t dimSize = x.size(actualDim);
    TORCH_CHECK(dimSize % kSplitFactor == 0, "x size at dim ", actualDim, " should be even, but got ", dimSize);
}
} // namespace

at::Tensor situ_glu_grad(const at::Tensor& grad_y, const at::Tensor& x, int64_t dim, double beta, double linear_beta,
                         bool activate_left)
{
    CheckNpuTensor(grad_y, "grad_y");
    CheckNpuTensor(x, "x");
    ValidateDim(x, dim);

    at::Tensor grad_x = at::empty(x.sizes(), x.options());

    ACLNN_CMD(aclnnSituGluGrad, grad_y, x, dim, beta, linear_beta, activate_left, grad_x);
    return grad_x;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_glu_grad", &cann_ops_nn::activation::situ_glu_grad, "SituGluGrad on NPU");
}
