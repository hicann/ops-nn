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
constexpr int64_t NUM_TWO = 2;

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

bool check_aclnn_kernel_available(std::string aclnn_name)
{
    std::string workspace_name = aclnn_name + "GetWorkspaceSize";
    if (GetOpApiFuncAddr(aclnn_name.c_str()) == nullptr || GetOpApiFuncAddr(workspace_name.c_str()) == nullptr) {
        return false;
    }
    return true;
}
} // namespace

at::Tensor clipped_swiglu(const at::Tensor& x, const c10::optional<at::Tensor>& group_index, int64_t dim, double alpha,
                          double limit, double bias, bool interleaved, int64_t clamp_mode)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(group_index, "group_index");
    TORCH_CHECK(clamp_mode == 0 || clamp_mode == 1, "clamp_mode should be 0 or 1, but got ", clamp_mode);

    if (dim < 0) {
        dim += static_cast<int64_t>(x.sizes().size());
    }
    TORCH_CHECK(dim >= 0 && dim < static_cast<int64_t>(x.sizes().size()), "dim out of range, got ", dim);
    TORCH_CHECK(x.size(dim) % NUM_TWO == 0, "x size at dim ", dim, " must be even, but got ", x.size(dim));
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    y_shape[dim] /= NUM_TWO;
    at::Tensor y = at::empty(y_shape, x.options());

    static bool npu_support_v2 = check_aclnn_kernel_available("aclnnClippedSwigluV2");
    if (npu_support_v2) {
        ACLNN_CMD(aclnnClippedSwigluV2, x, group_index, dim, alpha, limit, bias, interleaved, clamp_mode, y);
    } else {
        ACLNN_CMD(aclnnClippedSwiglu, x, group_index, dim, alpha, limit, bias, interleaved, y);
    }
    return y;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("clipped_swiglu", &cann_ops_nn::activation::clipped_swiglu, "ClippedSwiglu operator on NPU");
}
