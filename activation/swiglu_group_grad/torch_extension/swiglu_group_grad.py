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

SPLIT_NUM = 2


def _check_swiglu_group_backward_inputs(
    grad_output, x, weight, y_origin, group_index, clamp_limit
):
    torch._check(
        grad_output.dim() >= 1,
        lambda: f"grad_output must be at least 1D, but got {grad_output.dim()}D",
    )
    torch._check(
        grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32),
        lambda: (
            "grad_output must be FLOAT16, BFLOAT16, or FLOAT32, "
            f"but got {grad_output.dtype}"
        ),
    )
    torch._check(
        x.dim() >= 1,
        lambda: f"x must be at least 1D, but got {x.dim()}D",
    )
    torch._check(
        x.shape[-1] == grad_output.shape[-1] * SPLIT_NUM,
        lambda: "x.shape[-1] must equal 2 * grad_output.shape[-1]",
    )
    x_outer_numel = 1
    for s in x.shape[:-1]:
        x_outer_numel *= s
    grad_outer_numel = 1
    for s in grad_output.shape[:-1]:
        grad_outer_numel *= s
    torch._check(
        x_outer_numel == grad_outer_numel,
        lambda: f"x outer numel ({x_outer_numel}) must equal grad_output outer numel ({grad_outer_numel})",
    )
    torch._check(
        grad_output.shape[-1] > 0,
        lambda: "grad_output.shape[-1] must be greater than 0",
    )
    torch._check(
        x.dtype == grad_output.dtype, lambda: "x dtype must equal grad_output dtype"
    )
    torch._check(
        clamp_limit == -1.0 or clamp_limit > 0.0,
        lambda: f"clamp_limit must be -1.0 (no clamp) or > 0.0, but got {clamp_limit}",
    )

    if (weight is None) != (y_origin is None):
        raise RuntimeError("weight and y_origin must be provided together")
    if weight is not None:
        weight_element_num = weight.numel()
        total_rows = 1
        for s in grad_output.shape[:-1]:
            total_rows *= s
        torch._check(
            weight_element_num == total_rows,
            lambda: f"weight element num must equal total rows ({total_rows}), but got {weight_element_num}",
        )
        torch._check(
            weight.dtype == torch.float32, lambda: "weight dtype must be FLOAT"
        )
        torch._check(
            y_origin.dim() >= 1,
            lambda: "y_origin must be at least 1D",
        )
        torch._check(
            y_origin.shape[-1] == grad_output.shape[-1],
            lambda: "y_origin.shape[-1] must equal grad_output.shape[-1]",
        )
        y_origin_outer_numel = 1
        for s in y_origin.shape[:-1]:
            y_origin_outer_numel *= s
        torch._check(
            y_origin_outer_numel == total_rows,
            lambda: f"y_origin outer numel ({y_origin_outer_numel}) must equal total rows ({total_rows})",
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
        return [self.resolve_source("swiglu_group_grad.cpp")]

    def schema(self):
        return (
            "swiglu_group_backward(Tensor grad_output, Tensor x, *, Tensor? weight=None, "
            "Tensor? y_origin=None, Tensor? group_index=None, float clamp_limit=-1.0) "
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
            clamp_limit=-1.0,
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
    grad_output, x, *, weight=None, y_origin=None, group_index=None, clamp_limit=-1.0
):
    op_module = builder.load()
    return op_module.swiglu_group_backward(
        grad_output, x, weight, y_origin, group_index, clamp_limit
    )
