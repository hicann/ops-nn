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
void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_loss(const at::Tensor& hidden,
                                                                      const at::Tensor& weight,
                                                                      const at::Tensor& labels, double alpha,
                                                                      int64_t ignore_index)
{
    CheckNpuTensor(hidden, "hidden");
    CheckNpuTensor(weight, "weight");
    CheckNpuTensor(labels, "labels");

    TORCH_CHECK(hidden.dim() == 2, "hidden must be 2-D [M, H], but got dim=", hidden.dim());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D [V, H], but got dim=", weight.dim());
    TORCH_CHECK(labels.dim() == 1, "labels must be 1-D [M], but got dim=", labels.dim());
    TORCH_CHECK(hidden.size(0) == labels.size(0), "hidden.size(0) must equal labels.size(0), but got ", hidden.size(0),
                " vs ", labels.size(0));
    TORCH_CHECK(hidden.size(1) == weight.size(1), "hidden.size(1) must equal weight.size(1), but got ", hidden.size(1),
                " vs ", weight.size(1));
    TORCH_CHECK(hidden.scalar_type() == weight.scalar_type(), "hidden and weight dtype must match, but got ",
                hidden.scalar_type(), " vs ", weight.scalar_type());
    TORCH_CHECK(hidden.scalar_type() == at::kHalf || hidden.scalar_type() == at::kBFloat16,
                "hidden dtype must be float16 or bfloat16, but got ", hidden.scalar_type());
    TORCH_CHECK(labels.scalar_type() == at::kLong, "labels dtype must be int64, but got ", labels.scalar_type());
    TORCH_CHECK(alpha > 0.0, "alpha must be > 0, but got ", alpha);

    const auto m = hidden.size(0);
    at::Tensor loss = at::empty({}, hidden.options().dtype(at::kFloat));
    at::Tensor max_out = at::empty({m}, hidden.options().dtype(at::kFloat));
    at::Tensor sum_out = at::empty({m}, hidden.options().dtype(at::kFloat));
    at::Tensor d_h = at::empty(hidden.sizes(), hidden.options());

    ACLNN_CMD(aclnnChunkLoss, hidden, weight, labels, alpha, ignore_index, loss, max_out, sum_out, d_h);

    return std::make_tuple(std::move(loss), std::move(max_out), std::move(sum_out), std::move(d_h));
}

} // namespace loss
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("chunk_loss", &cann_ops_nn::loss::chunk_loss, "ChunkLoss on NPU"); }
