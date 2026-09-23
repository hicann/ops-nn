/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>

#include <torch/extension.h>
#include "aclnnop/aclnn_turbo_quant.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

constexpr int64_t N_CENT = 16;
constexpr int64_t SUPPORTED_HEAD_DIM = 512;

std::tuple<at::Tensor, at::Tensor> turbo_quant(const at::Tensor& latent, const at::Tensor& centroids)
{
    TORCH_CHECK(latent.device().type() == at::kPrivateUse1, "latent must be on NPU device");
    TORCH_CHECK(centroids.device().type() == at::kPrivateUse1, "centroids must be on NPU device");

    TORCH_CHECK(latent.dim() == 2, "latent must be 2-dimensional [numTokens, headDim], but got ", latent.dim());
    TORCH_CHECK(latent.scalar_type() == at::kFloat,
                "latent dtype must be float32; float16 and bfloat16 are unsupported, but got ", latent.scalar_type());
    TORCH_CHECK(latent.is_contiguous(), "latent must be contiguous");
    TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids dtype must be float32, but got ",
                centroids.scalar_type());
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
    TORCH_CHECK(centroids.dim() == 1 && centroids.size(0) == N_CENT, "centroids must have shape [", N_CENT,
                "], but got ", centroids.sizes());

    const int64_t num_tokens = latent.size(0);
    const int64_t head_dim = latent.size(1);
    TORCH_CHECK(head_dim == SUPPORTED_HEAD_DIM, "headDim only supports ", SUPPORTED_HEAD_DIM, " for now, but got ",
                head_dim);

    at::Tensor y = at::empty({num_tokens, head_dim / 2}, at::TensorOptions().dtype(at::kByte).device(latent.device()));
    at::Tensor scale = at::empty({num_tokens}, at::TensorOptions().dtype(at::kHalf).device(latent.device()));

    ACLNN_CMD(aclnnTurboQuant, latent, centroids, y, scale);
    return std::make_tuple(y, scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("turbo_quant", &cann_ops_nn::quant::turbo_quant, "TurboQuant 4-bit latent quantization on NPU");
}
