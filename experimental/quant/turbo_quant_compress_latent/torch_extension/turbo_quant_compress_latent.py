# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

N_CENT = 16
SCALE_BYTES = 2
SLOT_ALIGN = 64
SUPPORTED_HEAD_DIM = 512
OUTPUT_PADDED = 0
OUTPUT_COMPACT_CORRECTED = 1


def slot_size(head_dim, output_mode=OUTPUT_PADDED):
    """Bytes per compressed slot: packed nibbles + fp16 norm, padded up to SLOT_ALIGN."""
    used = head_dim // 2 + SCALE_BYTES
    if output_mode == OUTPUT_COMPACT_CORRECTED:
        return used
    return (used + SLOT_ALIGN - 1) // SLOT_ALIGN * SLOT_ALIGN


class TurboQuantCompressLatentOpBuilder(OpBuilder):
    """
    TurboQuantCompressLatent 算子的构建器

    基于 aclnnTurboQuantCompressLatent API 实现，将 MLA KV latent 逐 token 归一化后按 16 个码本中心
    量化成 4bit，两个 nibble 打包成一个字节，并在其后附加 float16 的 L2 范数。
    """

    def __init__(self):
        super().__init__("turbo_quant_compress_latent")

    def sources(self) -> list:
        return [self.resolve_source("turbo_quant_compress_latent.cpp")]

    def schema(self) -> str:
        return "turbo_quant_compress_latent(Tensor latent, Tensor centroids, int output_mode=0) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def turbo_quant_compress_latent_meta(
            latent: torch.Tensor,
            centroids: torch.Tensor,
            output_mode: int = OUTPUT_PADDED,
        ) -> torch.Tensor:
            torch._check(
                latent.dim() == 2,
                lambda: f"latent must be 2-dimensional [numTokens, headDim], but got {latent.dim()}-d",
            )
            torch._check(
                latent.dtype == torch.float32,
                lambda: f"latent dtype must be float32, but got {latent.dtype}",
            )
            torch._check(
                centroids.dtype == torch.float32,
                lambda: f"centroids dtype must be float32, but got {centroids.dtype}",
            )
            torch._check(
                centroids.numel() == N_CENT,
                lambda: f"centroids must hold exactly {N_CENT} elements, but got {centroids.numel()}",
            )
            num_tokens, head_dim = latent.shape
            torch._check(
                head_dim == SUPPORTED_HEAD_DIM,
                lambda: f"headDim only supports {SUPPORTED_HEAD_DIM} for now, but got {head_dim}",
            )
            torch._check(
                output_mode in (OUTPUT_PADDED, OUTPUT_COMPACT_CORRECTED),
                lambda: f"output_mode only supports 0 or 1, but got {output_mode}",
            )
            return torch.empty(
                (num_tokens, slot_size(head_dim, output_mode)),
                dtype=torch.uint8,
                device="meta",
            )


builder = TurboQuantCompressLatentOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def turbo_quant_compress_latent(
    latent: torch.Tensor, centroids: torch.Tensor, output_mode: int = OUTPUT_PADDED
) -> torch.Tensor:
    """
    NPU 上的 TurboQuant latent 压缩

    :param latent: [numTokens, headDim] float32，已完成 signed Hadamard 旋转且未归一化
    :param centroids: [16] float32，升序排列的 Lloyd-Max 码本
    :param output_mode: 0 生成 64B 对齐的兼容布局；1 生成 DS V4 使用的 258B compact corrected 布局
    :return: [numTokens, slot_size(headDim, output_mode)] uint8
    """
    op_module = builder.load()
    return op_module.turbo_quant_compress_latent(latent, centroids, output_mode)
