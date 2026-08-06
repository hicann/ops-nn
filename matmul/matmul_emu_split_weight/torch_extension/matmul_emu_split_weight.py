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


class MatmulEmuSplitWeightOpBuilder(OpBuilder):
    """
    MatmulEmuSplitWeight算子的构建器
    """

    def __init__(self):
        super().__init__("matmul_emu_split_weight", category="matmul")

    def sources(self) -> list:
        return [self.resolve_source("matmul_emu_split_weight.cpp")]

    def schema(self) -> str:
        return (
            "matmul_emu_split_weight("
            "Tensor x, "
            "Tensor w_high, "
            "Tensor w_low, "
            "*, "
            "float w_low_scale=0.00390625, "
            "int y_dtype=0"
            ") -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def matmul_emu_split_weight_meta(
            x: torch.Tensor,
            w_high: torch.Tensor,
            w_low: torch.Tensor,
            *,
            w_low_scale: float = 0.00390625,
            y_dtype: int = 0,
        ):
            torch._check(
                x.dim() == 2, lambda: f"The input x should be 2D, but got {x.dim()}D."
            )
            torch._check(
                w_high.dim() == 2,
                lambda: f"The input w_high should be 2D, but got {w_high.dim()}D.",
            )
            torch._check(
                w_low.dim() == 2,
                lambda: f"The input w_low should be 2D, but got {w_low.dim()}D.",
            )
            torch._check(
                w_high.size() == w_low.size(),
                lambda: f"w_low shape must match w_high shape, but got {w_low.size()} vs {w_high.size()}.",
            )
            torch._check(
                y_dtype == 0,
                lambda: f"y_dtype only supports 0(FP32), but got {y_dtype}.",
            )
            out_dtype = torch.float32
            return torch.empty(
                (x.size(0), w_high.size(1)), dtype=out_dtype, device="meta"
            )


matmul_emu_split_weight_builder = MatmulEmuSplitWeightOpBuilder()
op_module_matmul = matmul_emu_split_weight_builder.load()


@impl(
    get_as_library(),
    matmul_emu_split_weight_builder.name,
    "PrivateUse1",
)
def matmul_emu_split_weight(
    x: torch.Tensor,
    w_high: torch.Tensor,
    w_low: torch.Tensor,
    *,
    w_low_scale: float = 0.00390625,
    y_dtype: int = 0,
):
    """
    NPU设备上的MatmulEmuSplitWeight实现

    计算公式: y = X @ W_high + w_low_scale * (X @ W_low)
    """
    return op_module_matmul.matmul_emu_split_weight(
        x, w_high, w_low, w_low_scale, y_dtype
    )
