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
namespace loss {

namespace {
void CheckNpuTensor(const at::Tensor& tensor, const char* name, const at::Device& device)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on an NPU device");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same device as hidden");
}
} // namespace

at::Tensor chunk_loss_grad(const at::Tensor& hidden, const at::Tensor& weight, const at::Tensor& labels, double alpha,
                           const at::Tensor& softmax_max, const at::Tensor& softmax_sum, const at::Tensor& grad_output,
                           int64_t ignore_index)
{
    TORCH_CHECK(hidden.defined(), "hidden must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(hidden), "hidden must be on an NPU device");
    const auto device = hidden.device();
    CheckNpuTensor(weight, "weight", device);
    CheckNpuTensor(labels, "labels", device);
    CheckNpuTensor(softmax_max, "softmax_max", device);
    CheckNpuTensor(softmax_sum, "softmax_sum", device);
    CheckNpuTensor(grad_output, "grad_output", device);

    const c10::OptionalDeviceGuard device_guard(device);
    at::Tensor grad_weight = at::empty(weight.sizes(), weight.options());
    ACLNN_CMD(aclnnChunkLossBackward, hidden, weight, labels, alpha, softmax_max, softmax_sum, grad_output,
              ignore_index, grad_weight);
    return grad_weight;
}

} // namespace loss
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_loss_grad", &cann_ops_nn::loss::chunk_loss_grad, "ChunkLossGrad on NPU");
}
