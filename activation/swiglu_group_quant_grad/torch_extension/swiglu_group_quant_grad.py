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


class SwigluGroupQuantBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_group_quant_backward")

    def sources(self):
        return [self.resolve_source("swiglu_group_quant_grad.cpp")]

    def schema(self):
        return (
            "swiglu_group_quant_backward(Tensor grad_y, Tensor x, *, "
            "Tensor? weight=None, Tensor? y_origin=None, Tensor? group_index=None, "
            "float clamp_limit=-1.0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_group_quant_backward_meta(
            grad_y, x, *, weight=None, y_origin=None, group_index=None, clamp_limit=-1.0
        ):
            grad_x = x.new_empty(x.shape)
            if weight is not None:
                grad_weight = x.new_empty(weight.shape, dtype=torch.float32)
            else:
                grad_weight = x.new_empty([0], dtype=torch.float32)
            return grad_x, grad_weight


builder = SwigluGroupQuantBackwardOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _swiglu_group_quant_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    weight: torch.Tensor = None,
    y_origin: torch.Tensor = None,
    group_index: torch.Tensor = None,
    clamp_limit: float = -1.0,
):
    op_module = builder.load()
    return op_module.swiglu_group_quant_backward(
        grad_y, x, weight, y_origin, group_index, clamp_limit
    )


def swiglu_group_quant_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    weight: torch.Tensor = None,
    y_origin: torch.Tensor = None,
    group_index: torch.Tensor = None,
    clamp_limit: float = -1.0,
):
    return torch.ops.cann_ops_nn.swiglu_group_quant_backward(
        grad_y,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=clamp_limit,
    )
