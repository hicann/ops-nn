# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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


class SituMxQuantOpBuilder(OpBuilder):
    """
    SituMxQuant 算子的构建器

    基于 aclnnSituMxQuant API 实现，融合 Situ 激活函数和 MX 量化 (Microscaling Quantization)。
    输入为 FP16/BF16，输出为 FP8 (E4M3FN/E5M2) + E8M0 scale。
    """

    def __init__(self):
        super().__init__("situ_mx_quant")

    def sources(self) -> list:
        return ["csrc/quant/situ_mx_quant.cpp"]

    def schema(self) -> str:
        return (
            "situ_mx_quant("
            "Tensor x, "
            "float beta=1.0, "
            "float linear_beta=0.0, "
            "bool activate_left=False, "
            "int dst_type=36, "
            "str round_mode='rint'"
            ") -> (Tensor y, Tensor y_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def situ_mx_quant_meta(
            x: torch.Tensor,
            beta: float = 1.0,
            linear_beta: float = 0.0,
            activate_left: bool = False,
            dst_type: int = 36,
            round_mode: str = "rint",
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                x.dim() >= 1,
                lambda: f"x must be at least 1-dimensional, but got {x.dim()}-d",
            )
            torch._check(
                x.shape[-1] % 2 == 0,
                lambda: f"x last dim must be even, but got {x.shape[-1]}",
            )
            torch._check(
                x.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
            )
            torch._check(
                beta > 0.0, lambda: f"beta must be greater than 0, but got {beta}"
            )
            torch._check(
                dst_type in (35, 36),
                lambda: f"dst_type must be 36(E4M3FN) or 35(E5M2), but got {dst_type}",
            )
            torch._check(
                round_mode in ("rint", "round", "floor"),
                lambda: (
                    f"round_mode must be 'rint', 'round' or 'floor', but got {round_mode}"
                ),
            )
            torch._check(
                round_mode == "rint",
                lambda: f"FP8 output requires round_mode='rint', but got {round_mode}",
            )

            y_shape = list(x.shape)
            y_shape[-1] = x.shape[-1] // 2

            block_size = 32
            align_num = 2
            y_axis_size = (y_shape[-1] + align_num * block_size - 1) // (
                align_num * block_size
            )
            y_scale_shape = y_shape[:-1] + [y_axis_size, align_num]

            if dst_type == 35:
                y_dtype = torch.float8_e5m2
            else:
                y_dtype = torch.float8_e4m3fn
            y = torch.empty(y_shape, dtype=y_dtype, device="meta")
            y_scale = torch.empty(
                y_scale_shape, dtype=torch.float8_e8m0fnu, device="meta"
            )
            return y, y_scale


situ_mx_quant_builder = SituMxQuantOpBuilder()
situ_mx_quant_builder._ensure_initialized()


@impl(get_as_library(), situ_mx_quant_builder.name, "PrivateUse1")
def situ_mx_quant(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = False,
    dst_type: int = 36,
    round_mode: str = "rint",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 SituMxQuant — 融合 Situ 激活和 MX 量化
    """
    op_module = situ_mx_quant_builder.load()
    return op_module.situ_mx_quant(
        x, beta, linear_beta, activate_left, dst_type, round_mode
    )
