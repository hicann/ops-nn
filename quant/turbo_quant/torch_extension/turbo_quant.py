# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

N_CENT = 16
SUPPORTED_HEAD_DIM = 512


class TurboQuantOpBuilder(OpBuilder):
    """
    TurboQuant 算子的构建器

    基于 aclnnTurboQuant API 实现，将 MLA KV latent 逐 token 归一化后按 16 个码本中心
    量化成 4bit，两个 nibble 打包成一个字节，并在其后附加 float16 的 L2 范数。
    """

    def __init__(self):
        super().__init__("turbo_quant")

    def sources(self) -> list:
        return [self.resolve_source("turbo_quant.cpp")]

    def schema(self) -> str:
        return (
            "turbo_quant(Tensor latent, Tensor centroids) -> (Tensor y, Tensor scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def turbo_quant_meta(
            latent: torch.Tensor, centroids: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                latent.dim() == 2,
                lambda: f"latent must be 2-dimensional [numTokens, headDim], but got {latent.dim()}-d",
            )
            torch._check(
                latent.dtype == torch.float32,
                lambda: f"latent dtype must be float32; float16 and bfloat16 are unsupported, but got {latent.dtype}",
            )
            torch._check(
                latent.is_contiguous(),
                lambda: "latent must be contiguous",
            )
            torch._check(
                centroids.dtype == torch.float32,
                lambda: f"centroids dtype must be float32, but got {centroids.dtype}",
            )
            torch._check(
                centroids.is_contiguous(),
                lambda: "centroids must be contiguous",
            )
            torch._check(
                centroids.dim() == 1,
                lambda: f"centroids must have shape [{N_CENT}], but got {tuple(centroids.shape)}",
            )
            torch._check(
                centroids.size(0) == N_CENT,
                lambda: f"centroids must have shape [{N_CENT}], but got {tuple(centroids.shape)}",
            )
            num_tokens, head_dim = latent.shape
            torch._check(
                head_dim == SUPPORTED_HEAD_DIM,
                lambda: f"headDim only supports {SUPPORTED_HEAD_DIM} for now, but got {head_dim}",
            )
            y = torch.empty(
                (num_tokens, head_dim // 2), dtype=torch.uint8, device="meta"
            )
            scale = torch.empty((num_tokens,), dtype=torch.float16, device="meta")
            return y, scale


builder = TurboQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def turbo_quant(
    latent: torch.Tensor, centroids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 TurboQuant latent 压缩

    :param latent: [numTokens, headDim] float32（不支持 float16/bfloat16），已完成 signed Hadamard 旋转且未归一化
    :param centroids: [16] float32，升序排列的 Lloyd-Max 码本
    :return: ([numTokens, headDim // 2] uint8, [numTokens] float16)
    """
    op_module = builder.load()
    return op_module.turbo_quant(latent, centroids)
