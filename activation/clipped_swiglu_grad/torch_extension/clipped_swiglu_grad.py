# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class ClippedSwigluGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("clipped_swiglu_grad")

    def sources(self):
        return ["csrc/activation/clipped_swiglu_grad.cpp"]

    def schema(self):
        return (
            "clipped_swiglu_grad(Tensor grad_y, Tensor x, "
            "Tensor? group_index=None, int dim=-1, float alpha=1.702, "
            "float limit=7.0, float bias=1.0, bool interleaved=True, "
            "int clamp_mode=0) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def clipped_swiglu_grad_meta(
            grad_y,
            x,
            *,
            group_index=None,
            dim=-1,
            alpha=1.702,
            limit=7.0,
            bias=1.0,
            interleaved=True,
            clamp_mode=0,
        ):
            grad_x = x.new_empty(x.shape)
            return grad_x


builder = ClippedSwigluGradOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _clipped_swiglu_grad(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    group_index: torch.Tensor = None,
    dim: int = -1,
    alpha: float = 1.702,
    limit: float = 7.0,
    bias: float = 1.0,
    interleaved: bool = True,
    clamp_mode: int = 0,
):
    op_module = builder.load()
    return op_module.clipped_swiglu_grad(
        grad_y, x, group_index, dim, alpha, limit, bias, interleaved, clamp_mode
    )


def clipped_swiglu_grad(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    group_index: torch.Tensor = None,
    dim: int = -1,
    alpha: float = 1.702,
    limit: float = 7.0,
    bias: float = 1.0,
    interleaved: bool = True,
    clamp_mode: int = 0,
):
    return torch.ops.cann_ops_nn.clipped_swiglu_grad(
        grad_y,
        x,
        group_index=group_index,
        dim=dim,
        alpha=alpha,
        limit=limit,
        bias=bias,
        interleaved=interleaved,
        clamp_mode=clamp_mode,
    )
