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


class RmsNormDynamicQuantOpBuilder(OpBuilder):
    """
    RmsNormDynamicQuant 算子的构建器

    基于 aclnnRmsNormDynamicQuant API 实现，融合 RMS Normalization 与 INT8 动态量化，
    """

    def __init__(self):
        super().__init__("rms_norm_dynamic_quant")

    def sources(self) -> list:
        return ["csrc/norm/rms_norm_dynamic_quant.cpp"]

    def schema(self) -> str:
        return (
            "rms_norm_dynamic_quant("
            "Tensor x, "
            "Tensor gamma, "
            "Tensor? smooth_scales=None, "
            "Tensor? beta=None, "
            "float epsilon=1e-6, "
            "int dst_type=2"
            ") -> (Tensor y, Tensor scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def rms_norm_dynamic_quant_meta(
            x: torch.Tensor,
            gamma: torch.Tensor,
            smooth_scales: Optional[torch.Tensor] = None,
            beta: Optional[torch.Tensor] = None,
            epsilon: float = 1e-6,
            dst_type: int = 2,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                x.dim() >= 2 and x.dim() <= 8,
                lambda: f"x must be 2-8 dimensional, but got {x.dim()}-d",
            )
            torch._check(
                gamma.dim() == 1,
                lambda: f"gamma must be 1-dimensional, but got {gamma.dim()}-d",
            )
            torch._check(
                gamma.shape[0] == x.shape[-1],
                lambda: f"gamma size mismatch: {gamma.shape[0]} vs {x.shape[-1]}",
            )
            torch._check(
                x.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
            )
            torch._check(
                gamma.dtype == x.dtype,
                lambda: f"gamma dtype mismatch: {x.dtype} vs {gamma.dtype}",
            )

            scale_shape = x.shape[:-1]
            y = torch.empty(x.shape, dtype=torch.int8, device="meta")
            scale = torch.empty(scale_shape, dtype=torch.float32, device="meta")
            return y, scale


rms_norm_dynamic_quant_builder = RmsNormDynamicQuantOpBuilder()
op_module_norm = rms_norm_dynamic_quant_builder.load()


@impl(get_as_library(), rms_norm_dynamic_quant_builder.name, "PrivateUse1")
def rms_norm_dynamic_quant(
    x: torch.Tensor,
    gamma: torch.Tensor,
    smooth_scales: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    dst_type: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 RmsNormDynamicQuant — 融合 RMS Normalization 与 INT8 动态量化
    """
    return op_module_norm.rms_norm_dynamic_quant(
        x, gamma, smooth_scales, beta, epsilon, dst_type
    )
