# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class DequantSituQuantOpBuilder(OpBuilder):
    """
    DequantSituQuant 算子的构建器

    基于 aclnnDequantSituQuant API 实现，融合反量化 (Dequant)、Situ 激活函数和量化 (Quant) 三个操作。
    支持 INT8 (per-channel dequant + static/dynamic quant)、INT32 (MoE grouped-matmul + dynamic quant)
    和 BF16 (pre-dequantized + dynamic quant) 三种输入路径。
    """

    def __init__(self):
        super().__init__("dequant_situ_quant")

    def sources(self) -> list:
        return ["csrc/quant/dequant_situ_quant.cpp"]

    def schema(self) -> str:
        return (
            "dequant_situ_quant("
            "Tensor x, "
            "*, Tensor? weight_scale=None, "
            "Tensor? activation_scale=None, "
            "Tensor? bias=None, "
            "Tensor? quant_scale=None, "
            "Tensor? quant_offset=None, "
            "Tensor? group_index=None, "
            "float beta=4.0, "
            "float linear_beta=25.0, "
            "bool activate_left=True, "
            'str quant_type="dynamic"'
            ") -> (Tensor y, Tensor y_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dequant_situ_quant_meta(
            x: torch.Tensor,
            *,
            weight_scale: Optional[torch.Tensor] = None,
            activation_scale: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            quant_scale: Optional[torch.Tensor] = None,
            quant_offset: Optional[torch.Tensor] = None,
            group_index: Optional[torch.Tensor] = None,
            beta: float = 4.0,
            linear_beta: float = 25.0,
            activate_left: bool = True,
            quant_type: str = "dynamic",
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            is_int8 = x.dtype == torch.int8
            is_int32 = x.dtype == torch.int32
            is_bf16 = x.dtype == torch.bfloat16
            is_fp16 = x.dtype == torch.float16
            torch._check(
                is_int8 or is_int32 or is_bf16 or is_fp16,
                lambda: (
                    f"x dtype must be int8, int32, bfloat16, or float16, but got {x.dtype}"
                ),
            )

            if is_int8:
                torch._check(
                    x.dim() >= 2,
                    lambda: (
                        f"x must be at least 2-dimensional for int8, but got {x.dim()}-d"
                    ),
                )
            else:
                torch._check(
                    x.dim() == 2,
                    lambda: (
                        f"x must be 2-dimensional for int32/bfloat16, but got {x.dim()}-d"
                    ),
                )

            torch._check(
                x.shape[-1] % 2 == 0,
                lambda: f"x last dim must be even, but got {x.shape[-1]}",
            )

            y_shape = list(x.shape)
            y_shape[-1] = x.shape[-1] // 2
            if is_int8:
                scale_shape = list(x.shape[:-1])
            else:
                scale_shape = [x.shape[0]]
            y = torch.empty(y_shape, dtype=torch.int8, device="meta")
            scale = torch.empty(scale_shape, dtype=torch.float32, device="meta")
            return y, scale


dequant_situ_quant_builder = DequantSituQuantOpBuilder()
dequant_situ_quant_builder._ensure_initialized()


@impl(get_as_library(), dequant_situ_quant_builder.name, "PrivateUse1")
def dequant_situ_quant(
    x: torch.Tensor,
    *,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    quant_offset: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
    activate_left: bool = True,
    quant_type: str = "dynamic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 DequantSituQuant — 融合反量化、Situ 激活和量化
    """
    op_module = dequant_situ_quant_builder.load()
    return op_module.dequant_situ_quant(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        beta,
        linear_beta,
        activate_left,
        quant_type,
    )
