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
#include "aclnnop/aclnn_turbo_quant_compress_latent.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

constexpr int64_t N_CENT = 16;
constexpr int64_t SCALE_BYTES = 2;
constexpr int64_t SLOT_ALIGN = 64;
constexpr int64_t SUPPORTED_HEAD_DIM = 512;

int64_t SlotSize(int64_t head_dim)
{
    int64_t used = head_dim / 2 + SCALE_BYTES;
    return (used + SLOT_ALIGN - 1) / SLOT_ALIGN * SLOT_ALIGN;
}

at::Tensor turbo_quant_compress_latent(const at::Tensor& latent, const at::Tensor& centroids)
{
    TORCH_CHECK(latent.device().type() == at::kPrivateUse1, "latent must be on NPU device");
    TORCH_CHECK(centroids.device().type() == at::kPrivateUse1, "centroids must be on NPU device");

    TORCH_CHECK(latent.dim() == 2, "latent must be 2-dimensional [numTokens, headDim], but got ", latent.dim());
    TORCH_CHECK(latent.scalar_type() == at::kFloat, "latent dtype must be float32, but got ", latent.scalar_type());
    TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids dtype must be float32, but got ",
                centroids.scalar_type());
    TORCH_CHECK(centroids.numel() == N_CENT, "centroids must hold exactly ", N_CENT, " elements, but got ",
                centroids.numel());

    const int64_t num_tokens = latent.size(0);
    const int64_t head_dim = latent.size(1);
    TORCH_CHECK(head_dim == SUPPORTED_HEAD_DIM, "headDim only supports ", SUPPORTED_HEAD_DIM, " for now, but got ",
                head_dim);

    at::Tensor slot = at::empty({num_tokens, SlotSize(head_dim)},
                                at::TensorOptions().dtype(at::kByte).device(latent.device()));

    ACLNN_CMD(aclnnTurboQuantCompressLatent, latent, centroids, slot);
    return slot;
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("turbo_quant_compress_latent", &cann_ops_nn::quant::turbo_quant_compress_latent,
          "TurboQuant 4-bit latent compression on NPU");
}
