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


class AvgPool3DGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("avg_pool3d_backward")

    def sources(self) -> list:
        return ["csrc/pooling/avg_pool3_d_grad.cpp"]

    def schema(self) -> str:
        return [
            "avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, "
            "bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
            "avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, "
            "int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, "
            "*, Tensor(a!) grad_input) -> Tensor(a!)",
        ]

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def avg_pool3d_backward_meta(
            grad_output: torch.Tensor,
            self: torch.Tensor,
            kernel_size: list,
            stride: list,
            padding: list,
            ceil_mode: bool,
            count_include_pad: bool,
            divisor_override: int = None,
        ):
            torch._check(
                grad_output.ndim >= 4,
                lambda: f"grad_output rank must be >= 4, got {grad_output.ndim}",
            )
            torch._check(
                self.ndim == grad_output.ndim,
                lambda: "self rank must equal grad_output rank",
            )
            return torch.empty(self.shape, dtype=grad_output.dtype, device="meta")

        @impl(get_as_library(), self.name + ".grad_input", "Meta")
        def avg_pool3d_backward_grad_input_meta(
            grad_output: torch.Tensor,
            self: torch.Tensor,
            kernel_size: list,
            stride: list,
            padding: list,
            ceil_mode: bool,
            count_include_pad: bool,
            divisor_override: int = None,
            grad_input: torch.Tensor = None,
        ):
            return grad_input


avg_pool3d_backward_builder = AvgPool3DGradOpBuilder()
avg_pool3d_backward_builder._ensure_initialized()


def avg_pool3d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size: list,
    stride: list,
    padding: list,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int = None,
):
    raise NotImplementedError(
        "avg_pool3d_backward eager direct call was removed; use torch graph mode (torch.compile)."
    )
