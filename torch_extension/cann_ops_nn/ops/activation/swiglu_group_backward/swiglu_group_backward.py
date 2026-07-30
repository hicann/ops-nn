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


def _check_swiglu_group_backward_inputs(
    grad_output, x, weight, y_origin, group_index, clamp_limit
):
    torch._check(
        grad_output.dim() in (2, 3),
        lambda: f"grad_output must be 2D or 3D, but got {grad_output.dim()}D",
    )
    torch._check(
        grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32),
        lambda: (
            "grad_output must be FLOAT16, BFLOAT16, or FLOAT32, "
            f"but got {grad_output.dtype}"
        ),
    )
    torch._check(
        x.dim() == grad_output.dim(), lambda: "x rank must equal grad_output rank"
    )
    for dim in range(grad_output.dim() - 1):
        torch._check(
            x.shape[dim] == grad_output.shape[dim],
            lambda dim=dim: (f"x.shape[{dim}] must equal grad_output.shape[{dim}]"),
        )
    torch._check(
        x.shape[-1] == grad_output.shape[-1] * 2,
        lambda: "x.shape[-1] must equal 2 * grad_output.shape[-1]",
    )
    torch._check(
        grad_output.shape[-1] > 0,
        lambda: "grad_output.shape[-1] must be greater than 0",
    )
    torch._check(
        x.dtype == grad_output.dtype, lambda: "x dtype must equal grad_output dtype"
    )
    torch._check(
        clamp_limit >= 0.0, lambda: f"clamp_limit must be >= 0.0, but got {clamp_limit}"
    )

    if (weight is None) != (y_origin is None):
        raise RuntimeError("weight and y_origin must be provided together")
    if weight is not None:
        torch._check(
            weight.dim() == grad_output.dim(),
            lambda: "weight rank must equal grad_output rank",
        )
        for dim in range(grad_output.dim() - 1):
            torch._check(
                weight.shape[dim] == grad_output.shape[dim],
                lambda dim=dim: (
                    f"weight.shape[{dim}] must equal grad_output.shape[{dim}]"
                ),
            )
        torch._check(weight.shape[-1] == 1, lambda: "weight.shape[-1] must be 1")
        torch._check(
            weight.dtype == torch.float32, lambda: "weight dtype must be FLOAT"
        )
        torch._check(
            y_origin.shape == grad_output.shape,
            lambda: "y_origin shape must equal grad_output shape",
        )
        torch._check(
            y_origin.dtype == grad_output.dtype,
            lambda: "y_origin dtype must equal grad_output dtype",
        )
    if group_index is not None:
        torch._check(
            group_index.dim() == 1, lambda: "group_index must be 1D when present"
        )
        torch._check(
            group_index.numel() > 0,
            lambda: "group_index must not be empty when present",
        )
        torch._check(
            group_index.dtype == torch.int64, lambda: "group_index dtype must be INT64"
        )


class SwigluGroupBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_group_backward")

    def sources(self):
        return ["csrc/activation/swiglu_group_backward.cpp"]

    def schema(self):
        return (
            "swiglu_group_backward(Tensor grad_output, Tensor x, *, Tensor? weight=None, "
            "Tensor? y_origin=None, Tensor? group_index=None, float clamp_limit=0.0) "
            "-> (Tensor, Tensor?)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_group_backward_meta(
            grad_output,
            x,
            *,
            weight=None,
            y_origin=None,
            group_index=None,
            clamp_limit=0.0,
        ):
            _check_swiglu_group_backward_inputs(
                grad_output, x, weight, y_origin, group_index, clamp_limit
            )
            grad_x = grad_output.new_empty(x.shape)
            if weight is not None:
                grad_weight = weight.new_empty(weight.shape, dtype=torch.float32)
            else:
                grad_weight = None
            return grad_x, grad_weight


builder = SwigluGroupBackwardOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def swiglu_group_backward(
    grad_output, x, *, weight=None, y_origin=None, group_index=None, clamp_limit=0.0
):
    op_module = builder.load()
    return op_module.swiglu_group_backward(
        grad_output, x, weight, y_origin, group_index, clamp_limit
    )
