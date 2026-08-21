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

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cross_entropy_sum_exp_and_index_logit(
    const at::Tensor& vocab_parallel_logits, const at::Tensor& target, const at::Tensor& global_logits_max,
    int64_t vocab_start_index, int64_t vocab_end_index)
{
    CheckNpuTensor(vocab_parallel_logits, "vocab_parallel_logits");
    CheckNpuTensor(target, "target");
    CheckNpuTensor(global_logits_max, "global_logits_max");

    auto target_shape = target.sizes();
    auto logits_shape = vocab_parallel_logits.sizes();

    // 浮点输出固定为 float32，整型输出固定为 int32（与 aclnnCrossEntropySumExpAndIndexLogit 约束一致）
    at::Tensor predicted_logits = at::empty(target_shape, vocab_parallel_logits.options().dtype(at::kFloat));
    at::Tensor sum_exp_logits = at::empty(target_shape, vocab_parallel_logits.options().dtype(at::kFloat));
    at::Tensor exp_logits = at::empty(logits_shape, vocab_parallel_logits.options().dtype(at::kFloat));
    at::Tensor target_offset = at::empty(target_shape, target.options().dtype(at::kInt));
    at::Tensor target_mask = at::empty(target_shape, target.options().dtype(at::kInt));

    ACLNN_CMD(aclnnCrossEntropySumExpAndIndexLogit, vocab_parallel_logits, target, global_logits_max, vocab_start_index,
              vocab_end_index, predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask);

    return std::make_tuple(std::move(predicted_logits), std::move(sum_exp_logits), std::move(exp_logits),
                           std::move(target_offset), std::move(target_mask));
}

} // namespace loss
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cross_entropy_sum_exp_and_index_logit", &cann_ops_nn::loss::cross_entropy_sum_exp_and_index_logit,
          "CrossEntropySumExpAndIndexLogit on NPU");
}
