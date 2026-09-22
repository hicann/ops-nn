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
#include <iostream>
#include <string>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace activation {
namespace {

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckSameDevice(const at::Tensor& ref, const at::Tensor& other, const char* ref_name, const char* other_name)
{
    TORCH_CHECK(ref.device() == other.device(), other_name, " must be on the same device as ", ref_name);
}

void CheckShapeMatch(const at::Tensor& ref, const at::Tensor& other, const char* ref_name, const char* other_name)
{
    TORCH_CHECK(other.sizes() == ref.sizes(), other_name, " shape must match ", ref_name);
}

// 调试用：打印 tensor 是否连续、shape 与 stride。
void PrintTensorLayout(const at::Tensor& tensor, const char* name)
{
    std::cout << "[ClaGateBackward] " << name << " is_contiguous=" << (tensor.is_contiguous() ? "true" : "false")
              << " shape=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        std::cout << (i > 0 ? ", " : "") << tensor.size(i);
    }
    std::cout << "] stride=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        std::cout << (i > 0 ? ", " : "") << tensor.stride(i);
    }
    std::cout << "] storage_offset=" << tensor.storage_offset() << std::endl;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cla_gate_backward(
    const at::Tensor& grad_merged, const at::Tensor& global_attn, const at::Tensor& local_attn,
    const at::Tensor& global_gate_logits, const at::Tensor& local_gate_logits, const std::string& input_attn_layout)
{
    CheckNpuTensor(grad_merged, "grad_merged");
    CheckNpuTensor(global_attn, "global_attn");
    CheckNpuTensor(local_attn, "local_attn");
    CheckNpuTensor(global_gate_logits, "global_gate_logits");
    CheckNpuTensor(local_gate_logits, "local_gate_logits");

    CheckSameDevice(global_attn, grad_merged, "global_attn", "grad_merged");
    CheckSameDevice(global_attn, local_attn, "global_attn", "local_attn");
    CheckSameDevice(global_attn, global_gate_logits, "global_attn", "global_gate_logits");
    CheckSameDevice(global_attn, local_gate_logits, "global_attn", "local_gate_logits");

    // 三路 TND 输入必须为 3D [T, N, D] 且 shape 一致
    TORCH_CHECK(grad_merged.dim() == 3, "grad_merged must be 3D [T,N,D], but got ", grad_merged.dim(), "D");
    TORCH_CHECK(global_attn.dim() == 3, "global_attn must be 3D [T,N,D], but got ", global_attn.dim(), "D");
    TORCH_CHECK(local_attn.dim() == 3, "local_attn must be 3D [T,N,D], but got ", local_attn.dim(), "D");
    CheckShapeMatch(global_attn, grad_merged, "global_attn", "grad_merged");
    CheckShapeMatch(global_attn, local_attn, "global_attn", "local_attn");

    const int64_t t = global_attn.size(0);
    const int64_t n = global_attn.size(1);

    // gate logits：2D [T, N]
    TORCH_CHECK(global_gate_logits.dim() == 2, "global_gate_logits must be 2D [T,N], but got ",
                global_gate_logits.dim(), "D");
    TORCH_CHECK(local_gate_logits.dim() == 2, "local_gate_logits must be 2D [T,N], but got ", local_gate_logits.dim(),
                "D");
    CheckShapeMatch(global_gate_logits, local_gate_logits, "global_gate_logits", "local_gate_logits");
    TORCH_CHECK(global_gate_logits.size(0) == t && global_gate_logits.size(1) == n, "gate logits must match [T,N]");

    // dtype 校验：5 路同 dtype（BF16/FP16）
    const auto input_dtype = global_attn.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "global_attn dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(grad_merged.scalar_type() == input_dtype, "grad_merged dtype must match global_attn dtype");
    TORCH_CHECK(local_attn.scalar_type() == input_dtype, "local_attn dtype must match global_attn dtype");
    TORCH_CHECK(global_gate_logits.scalar_type() == input_dtype,
                "global_gate_logits dtype must match global_attn dtype");
    TORCH_CHECK(local_gate_logits.scalar_type() == input_dtype, "local_gate_logits dtype must match global_attn dtype");

    // 不支持空 Tensor
    TORCH_CHECK(grad_merged.numel() > 0 && global_attn.numel() > 0 && local_attn.numel() > 0 &&
                    global_gate_logits.numel() > 0 && local_gate_logits.numel() > 0,
                "ClaGateBackward does not support empty tensor");

    TORCH_CHECK(input_attn_layout == "TND", "input_attn_layout only supports \"TND\", but got ", input_attn_layout);

    // 高进高出：输出与对应输入同 shape/dtype。
    at::Tensor grad_global_attn_out = at::empty(global_attn.sizes(), global_attn.options());
    at::Tensor grad_local_attn_out = at::empty(local_attn.sizes(), local_attn.options());
    at::Tensor grad_global_gate_logits_out = at::empty(global_gate_logits.sizes(), global_gate_logits.options());
    at::Tensor grad_local_gate_logits_out = at::empty(local_gate_logits.sizes(), local_gate_logits.options());

    ACLNN_CMD(aclnnClaGateBackward, grad_merged, global_attn, local_attn, global_gate_logits, local_gate_logits,
              input_attn_layout, grad_global_attn_out, grad_local_attn_out, grad_global_gate_logits_out,
              grad_local_gate_logits_out);

    return std::make_tuple(std::move(grad_global_attn_out), std::move(grad_local_attn_out),
                           std::move(grad_global_gate_logits_out), std::move(grad_local_gate_logits_out));
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cla_gate_backward", &cann_ops_nn::activation::cla_gate_backward, "ClaGateBackward operator on NPU");
}
