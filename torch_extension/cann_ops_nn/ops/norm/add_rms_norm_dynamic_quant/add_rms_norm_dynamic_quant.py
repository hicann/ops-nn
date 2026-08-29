# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class AddRmsNormDynamicQuantOpBuilder(OpBuilder):
    """
    AddRmsNormDynamicMxQuant 算子构建器

    基于 aclnnAddRmsNormDynamicMxQuantV2 API 实现，融合 Add + RMS Normalization + MX 动态量化。
    当 x3 提供时计算 x = (x3 + x1) + x2，否则 x = x1 + x2。
    始终走 V2 接口，x3 为 None 时内部传 nullptr，行为与 V1 一致。
    """

    def __init__(self):
        super().__init__("add_rms_norm_dynamic_quant")

    def sources(self) -> list:
        return [self.resolve_source("add_rms_norm_dynamic_quant.cpp")]

    def schema(self) -> str:
        return (
            "add_rms_norm_dynamic_quant("
            "Tensor x1, "
            "Tensor x2, "
            "Tensor gamma, "
            "Tensor? beta=None, "
            "Tensor? x3=None, "
            "float epsilon=1e-6, "
            "int scale_alg=0, "
            'str round_mode="rint", '
            "int dst_type=40, "
            "bool output_rstd=False"
            ") -> (Tensor y, Tensor x, Tensor mxscale, Tensor rstd)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def add_rms_norm_dynamic_quant_meta(
            x1: torch.Tensor,
            x2: torch.Tensor,
            gamma: torch.Tensor,
            beta: Optional[torch.Tensor] = None,
            x3: Optional[torch.Tensor] = None,
            epsilon: float = 1e-6,
            scale_alg: int = 0,
            round_mode: str = "rint",
            dst_type: int = 40,
            output_rstd: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            torch._check(
                x1.dim() >= 1 and x1.dim() <= 7,
                lambda: f"x1 must be 1-7 dimensional, but got {x1.dim()}-d",
            )
            torch._check(
                x1.shape == x2.shape,
                lambda: f"x1 and x2 must have the same shape, got {x1.shape} vs {x2.shape}",
            )
            torch._check(
                gamma.dim() == 1 and gamma.shape[0] == x1.shape[-1],
                lambda: f"gamma must be 1D with size matching x1 last dim ({x1.shape[-1]})",
            )
            torch._check(
                x1.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x1 dtype must be float16 or bfloat16, but got {x1.dtype}",
            )
            if beta is not None:
                torch._check(
                    beta.shape == gamma.shape,
                    lambda: f"beta must have the same shape as gamma, got {beta.shape} vs {gamma.shape}",
                )
                torch._check(
                    beta.dtype == gamma.dtype,
                    lambda: f"beta dtype must match gamma dtype, got {beta.dtype} vs {gamma.dtype}",
                )
            if x3 is not None:
                torch._check(
                    x3.shape == x1.shape,
                    lambda: f"x3 must have the same shape as x1, got {x3.shape} vs {x1.shape}",
                )
                torch._check(
                    x3.dtype == x1.dtype,
                    lambda: f"x3 dtype must match x1 dtype, got {x3.dtype} vs {x1.dtype}",
                )

            is_fp4 = dst_type in (40, 41)
            if is_fp4:
                torch._check(
                    x1.shape[-1] % 2 == 0,
                    lambda: f"x1 last dim must be even for FP4 dst_type, got {x1.shape[-1]}",
                )
                torch._check(
                    scale_alg == 0,
                    lambda: f"scale_alg must be 0 (OCP) for FP4 dst_type, got {scale_alg}",
                )

            y_shape = list(x1.shape)
            if is_fp4:
                y_dtype = torch.uint8
                y_shape[-1] //= 2
            elif dst_type == 35:
                y_dtype = torch.float8_e5m2
            elif dst_type == 36:
                y_dtype = torch.float8_e4m3fn
            else:
                torch._check(
                    False, lambda: f"invalid dst_type {dst_type}, expected 35/36/40/41"
                )

            num_blocks = (x1.shape[-1] + 31) // 32
            mxscale_last = (num_blocks + 1) // 2
            mxscale_shape = list(x1.shape)
            mxscale_shape[-1] = mxscale_last
            mxscale_shape.append(2)

            rstd_shape = list(x1.shape)
            rstd_shape[-1] = 1

            y = torch.empty(y_shape, dtype=y_dtype, device="meta")
            x_out = torch.empty(x1.shape, dtype=x1.dtype, device="meta")
            mxscale = torch.empty(
                mxscale_shape, dtype=torch.float8_e8m0fnu, device="meta"
            )
            rstd = torch.empty(rstd_shape, dtype=torch.float32, device="meta")
            return y, x_out, mxscale, rstd


add_rms_norm_dynamic_quant_builder = AddRmsNormDynamicQuantOpBuilder()
add_rms_norm_dynamic_quant_builder._ensure_initialized()


@impl(get_as_library(), add_rms_norm_dynamic_quant_builder.name, "PrivateUse1")
def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    x3: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = 40,
    output_rstd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NPU AddRmsNormDynamicMxQuant — Add + RmsNorm + MX 动态量化融合算子

    当 x3 提供时: x = (x3 + x1) + x2
    当 x3 为 None: x = x1 + x2
    """
    op_module = add_rms_norm_dynamic_quant_builder.load()
    return op_module.add_rms_norm_dynamic_quant(
        x1,
        x2,
        gamma,
        beta,
        x3,
        epsilon,
        scale_alg,
        round_mode,
        dst_type,
        output_rstd,
    )
