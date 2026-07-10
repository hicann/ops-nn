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

c10::SmallVector<int64_t, op_infer::SIZE> GetSwigluOutputShape(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    const int64_t lastDim = x.size(x.dim() - 1);
    TORCH_CHECK(lastDim % kSplitFactor == 0, "x last dim size should be even");

    auto yShape = op_infer::array_to_small_vector(x.sizes());
    yShape[x.dim() - 1] = lastDim / kSplitFactor;
    return yShape;
}
} // namespace

at::Tensor swiglu_group(const at::Tensor& x, const c10::optional<at::Tensor>& weight,
                        const c10::optional<at::Tensor>& group_index, double clamp_limit)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(group_index, "group_index");

    at::Tensor y = at::empty(GetSwigluOutputShape(x), x.options());

    ACLNN_CMD(aclnnSwigluGroup, x, weight, group_index, clamp_limit, y);
    return y;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group", &cann_ops_nn::activation::swiglu_group, "SwigluGroup on NPU");
}
