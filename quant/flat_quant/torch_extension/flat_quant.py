# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class FlatQuantOpBuilder(OpBuilder):
    """
    FlatQuant 算子的构建器

    基于 aclnnFlatQuantV3 API 实现，为矩阵x依次进行两次小矩阵乘法，然后针对矩阵乘的结果进行量化处理。
    """

    def __init__(self):
        super().__init__("flat_quant")

    def sources(self) -> list:
        return [self.resolve_source("flat_quant.cpp")]

    def schema(self) -> str:
        return (
            "flat_quant("
            "Tensor x, "
            "Tensor kronecker_p1, "
            "Tensor kronecker_p2, "
            "float clip_ratio=1.0, "
            "int dst_dtype=16, "
            "float dst_type_max=0.0, "
            "Tensor? group_list=None, "
            "int group_list_type=0"
            ") -> (Tensor out, Tensor quant_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def flat_quant_meta(
            x: torch.Tensor,
            kronecker_p1: torch.Tensor,
            kronecker_p2: torch.Tensor,
            clip_ratio: float = 1.0,
            dst_dtype: int = 16,
            dst_type_max: float = 0.0,
            group_list: Optional[torch.Tensor] = None,
            group_list_type: int = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # 维度检查
            torch._check(
                x.dim() == 3,
                lambda: f"x must be 3-dimensional [M, N1, N2], but got {x.dim()}-d",
            )
            M, N1, N2 = x.shape

            torch._check(
                kronecker_p1.dim() == 2,
                lambda: f"kronecker_p1 must be 2-dimensional [N1, N1], but got {kronecker_p1.dim()}-d",
            )
            torch._check(
                kronecker_p1.shape[0] == N1 and kronecker_p1.shape[1] == N1,
                lambda: f"kronecker_p1 shape must be [{N1}, {N1}], but got [{kronecker_p1.shape[0]}, {kronecker_p1.shape[1]}]",
            )

            torch._check(
                kronecker_p2.dim() == 2,
                lambda: f"kronecker_p2 must be 2-dimensional [N2, N2], but got {kronecker_p2.dim()}-d",
            )
            torch._check(
                kronecker_p2.shape[0] == N2 and kronecker_p2.shape[1] == N2,
                lambda: f"kronecker_p2 shape must be [{N2}, {N2}], but got [{kronecker_p2.shape[0]}, {kronecker_p2.shape[1]}]",
            )

            # 数据类型检查
            torch._check(
                x.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
            )
            torch._check(
                kronecker_p1.dtype == x.dtype,
                lambda: f"kronecker_p1 dtype mismatch: {x.dtype} vs {kronecker_p1.dtype}",
            )
            torch._check(
                kronecker_p2.dtype == x.dtype,
                lambda: f"kronecker_p2 dtype mismatch: {x.dtype} vs {kronecker_p2.dtype}",
            )

            # 参数范围检查
            torch._check(
                clip_ratio > 0.0 and clip_ratio <= 1.0,
                lambda: f"clip_ratio must be in range (0, 1], but got {clip_ratio}",
            )
            torch._check(
                dst_type_max == 0.0 or (dst_type_max >= 6.0 and dst_type_max <= 12.0),
                lambda: f"dst_type_max must be 0 or in range [6, 12], but got {dst_type_max}",
            )
            torch._check(
                group_list_type >= 0 and group_list_type <= 2,
                lambda: f"group_list_type must be in range [0, 2], but got {group_list_type}",
            )

            # 确定输出形状
            torch._check(
                dst_dtype == 16 or dst_dtype == 296,
                lambda: f"dst_dtype must be in range [16, 296], but got {dst_dtype}",
            )
            if dst_dtype == 16:
                torch._check(
                    N2 % 8 == 0,
                    lambda: f"N2 must be divisible by 8 for INT4 output, but got N2={N2}",
                )
                out_shape = (M, N1, N2 // 8)
                out_dtype = torch.int32
                quant_scale_shape = (M,)
                quant_scale_dtype = torch.float32
            else:
                torch._check(
                    N2 % 2 == 0,
                    lambda: f"N2 must be even for FP4 output, but got N2={N2}",
                )
                out_shape = (M, N1 * N2 // 2)
                out_dtype = torch.uint8
                quant_scale_shape = (M, (N1 * N2 + 63) // 64, 2)
                quant_scale_dtype = torch.uint8

            out = torch.empty(out_shape, dtype=out_dtype, device="meta")
            quant_scale = torch.empty(
                quant_scale_shape, dtype=quant_scale_dtype, device="meta"
            )
            return out, quant_scale


flat_quant_builder = FlatQuantOpBuilder()
flat_quant_builder._ensure_initialized()


@impl(get_as_library(), flat_quant_builder.name, "PrivateUse1")
def flat_quant(
    x: torch.Tensor,
    kronecker_p1: torch.Tensor,
    kronecker_p2: torch.Tensor,
    clip_ratio: float = 1.0,
    dst_dtype: int = 16,
    dst_type_max: float = 0.0,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 FlatQuant — 矩阵克罗内克乘积量化
    """
    op_module_quant = flat_quant_builder.load()
    return op_module_quant.flat_quant(
        x,
        kronecker_p1,
        kronecker_p2,
        clip_ratio,
        dst_dtype,
        dst_type_max,
        group_list,
        group_list_type,
    )
