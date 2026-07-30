#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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


class GroupedDynamicMxQuantWithDualAxisOpBuilder(OpBuilder):
    """
    GroupedDynamicMxQuantWithDualAxis算子的构建器
    """

    def __init__(self):
        super().__init__("grouped_dynamic_mx_quant_with_dual_axis", category="quant")

    def sources(self) -> list:
        return [self.resolve_source("grouped_dynamic_mx_quant_with_dual_axis.cpp")]

    def schema(self) -> str:
        return (
            "grouped_dynamic_mx_quant_with_dual_axis("
            "Tensor x, "
            "Tensor group_index, "
            "*, "
            'str round_mode="rint", '
            "int scale_alg=1, "
            "int dst_type=24, "
            "float dst_type_max=0"
            ") -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_dynamic_mx_quant_with_dual_axis_meta(
            x: torch.Tensor,
            group_index: torch.Tensor,
            *,
            round_mode: str = "rint",
            scale_alg: int = 1,
            dst_type: int = 24,
            dst_type_max: float = 0,
        ):
            torch._check(
                x.dim() == 2, lambda: f"The input x should be 2D, but got {x.dim()}D."
            )
            torch._check(
                group_index.dim() == 1 and group_index.numel() > 0,
                lambda: (
                    f"The input group_index should be a non-empty 1D tensor, "
                    f"but got {group_index.dim()}D with {group_index.numel()} elements."
                ),
            )
            torch._check(
                x.size(0) > 0,
                lambda: f"The first dim of input x should be positive, but got {x.size(0)}.",
            )
            torch._check(
                x.size(1) > 0,
                lambda: f"The last dim of input x should be positive, but got {x.size(1)}.",
            )
            torch._check(
                x.size(1) % 64 == 0,
                lambda: f"The last dim of input x should be divisible by 64, but got {x.size(1)}.",
            )
            torch._check(
                round_mode == "rint",
                lambda: f'The round_mode only supports "rint", but got "{round_mode}".',
            )
            torch._check(
                dst_type in (23, 24),
                lambda: (
                    "The dst_type should be 23 (torch.float8_e5m2) or "
                    f"24 (torch.float8_e4m3fn), but got {dst_type}."
                ),
            )

            if dst_type == 23:
                y1 = torch.empty_like(x, dtype=torch.float8_e5m2)
                y2 = torch.empty_like(x, dtype=torch.float8_e5m2)
            else:
                y1 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                y2 = torch.empty_like(x, dtype=torch.float8_e4m3fn)

            m = x.size(0)
            n = x.size(1)
            mxscale1 = x.new_empty([m, n // 64, 2], dtype=torch.uint8)
            mxscale2 = x.new_empty(
                [m // 64 + group_index.size(0), n, 2], dtype=torch.uint8
            )
            return (y1, mxscale1, y2, mxscale2)


grouped_dynamic_mx_quant_with_dual_axis_builder = (
    GroupedDynamicMxQuantWithDualAxisOpBuilder()
)
grouped_dynamic_mx_quant_with_dual_axis_builder._ensure_initialized()


@impl(
    get_as_library(),
    grouped_dynamic_mx_quant_with_dual_axis_builder.name,
    "PrivateUse1",
)
def grouped_dynamic_mx_quant_with_dual_axis(
    x: torch.Tensor,
    group_index: torch.Tensor,
    *,
    round_mode: str = "rint",
    scale_alg: int = 1,
    dst_type: int = 24,
    dst_type_max: float = 0,
):
    """
    NPU设备上的GroupedDynamicMxQuantWithDualAxis实现
    """
    op_module_quant = grouped_dynamic_mx_quant_with_dual_axis_builder.load()
    return op_module_quant.grouped_dynamic_mx_quant_with_dual_axis(
        x, group_index, round_mode, scale_alg, dst_type, dst_type_max
    )
