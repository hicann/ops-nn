/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace activation {
namespace {
constexpr int64_t SPLIT_FACTOR = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(tensor.value()), name, " must be on NPU device");
    }
}

void CheckSiTuGroupAttrs(const double beta, const double linearBeta)
{
    TORCH_CHECK(std::isfinite(beta) && beta != 0.0, "beta must be finite and non-zero, but got ", beta);
    TORCH_CHECK(std::isfinite(linearBeta), "linear_beta must be finite, but got ", linearBeta);
}

c10::SmallVector<int64_t, op_infer::SIZE> GetSituGluGroupOutputShape(const at::Tensor& x, const int64_t dim)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    int64_t actualDim = dim;
    if (actualDim < 0) {
        actualDim += x.dim();
    }
    TORCH_CHECK(actualDim >= 0 && actualDim < x.dim(), "dim should be in [", -x.dim(), ", ", x.dim() - 1, "], but got ",
                dim);
    const int64_t dimSize = x.size(actualDim);
    TORCH_CHECK(dimSize % SPLIT_FACTOR == 0, "x size at dim ", actualDim, " should be even, but got ", dimSize);

    auto yShape = op_infer::array_to_small_vector(x.sizes());
    yShape[actualDim] = dimSize / SPLIT_FACTOR;
    return yShape;
}
} // namespace

at::Tensor situ_glu_group(const at::Tensor& x, const c10::optional<at::Tensor>& weight,
                          const c10::optional<at::Tensor>& group_index, const int64_t dim, const double beta,
                          const double linear_beta, const bool activate_left)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(group_index, "group_index");
    CheckSiTuGroupAttrs(beta, linear_beta);

    at::Tensor y = at::empty(GetSituGluGroupOutputShape(x, dim), x.options());

    ACLNN_CMD(aclnnSituGluGroup, x, weight, group_index, dim, beta, linear_beta, activate_left, y);
    return y;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_glu_group", &cann_ops_nn::activation::situ_glu_group, "SituGluGroup on NPU");
}
