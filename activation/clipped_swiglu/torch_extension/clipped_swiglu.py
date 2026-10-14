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


class ClippedSwigluOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("clipped_swiglu")

    def sources(self) -> list:
        return [self.resolve_source("clipped_swiglu.cpp")]

    def schema(self) -> str:
        return (
            "clipped_swiglu("
            "Tensor x, Tensor? group_index=None, "
            "*, int dim=-1, float alpha=1.702, float limit=7.0, float bias=1.0, bool interleaved=True, int clamp_mode=0"
            ") -> Tensor "
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def clipped_swiglu_meta(
            x: torch.Tensor,
            group_index=None,
            *,
            dim=-1,
            alpha=1.702,
            limit=7.0,
            bias=1.0,
            interleaved=True,
            clamp_mode=0,
        ):
            real_dim = dim if dim >= 0 else dim + x.dim()
            if x.dim() == 0:
                raise RuntimeError("x must be at least 1-D")
            if real_dim < 0 or real_dim >= x.dim():
                raise RuntimeError(
                    f"dim out of range [-{x.dim()}, {x.dim() - 1}], got {dim}"
                )
            if x.size(real_dim) % 2 != 0:
                raise RuntimeError(
                    f"x size at dim {real_dim} must be even, but got {x.size(real_dim)}"
                )
            output_size = list(x.shape)
            output_size[real_dim] = output_size[real_dim] // 2
            return torch.empty(output_size, dtype=x.dtype, device=x.device)


clipped_swiglu_builder = ClippedSwigluOpBuilder()
clipped_swiglu_builder._ensure_initialized()


@impl(get_as_library(), clipped_swiglu_builder.name, "PrivateUse1")
def clipped_swiglu(
    x: torch.Tensor,
    group_index=None,
    *,
    dim=-1,
    alpha=1.702,
    limit=7.0,
    bias=1.0,
    interleaved=True,
    clamp_mode=0,
):
    op_module = clipped_swiglu_builder.load()
    return op_module.clipped_swiglu(
        x, group_index, dim, alpha, limit, bias, interleaved, clamp_mode
    )


def _clipped_swiglu_backward_autograd(ctx, grad_y):
    saved_x, saved_group_index = ctx.saved_tensors
    if ctx.clamp_mode != 0:
        raise RuntimeError(
            "ClippedSwigluGrad (V1) only supports clamp_mode=0, but got "
            f"{ctx.clamp_mode}. The backward of clipped_swiglu with clamp_mode!=0 "
            "is not supported yet."
        )
    return (
        torch.ops.cann_ops_nn.clipped_swiglu_grad(
            grad_y,
            saved_x,
            group_index=saved_group_index,
            dim=ctx.dim,
            alpha=ctx.alpha,
            limit=ctx.limit,
            bias=ctx.bias,
            interleaved=ctx.interleaved,
            clamp_mode=ctx.clamp_mode,
        ),
        None,
    )


def _clipped_swiglu_setup_context(ctx, inputs, keyword_only_inputs, output):
    (x, group_index) = inputs
    ctx.save_for_backward(x, group_index)
    ctx.dim = keyword_only_inputs["dim"]
    ctx.alpha = keyword_only_inputs["alpha"]
    ctx.limit = keyword_only_inputs["limit"]
    ctx.bias = keyword_only_inputs["bias"]
    ctx.interleaved = keyword_only_inputs["interleaved"]
    ctx.clamp_mode = keyword_only_inputs["clamp_mode"]


torch.library.register_autograd(
    "cann_ops_nn::clipped_swiglu",
    _clipped_swiglu_backward_autograd,
    setup_context=_clipped_swiglu_setup_context,
    lib=get_as_library(),
)
