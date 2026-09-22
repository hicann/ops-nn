# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Optional

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


FP8_E5M2 = 35
FP8_E4M3FN = 36


class SwigluBackwardGroupQuantWithDualAxisOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_backward_group_quant_with_dual_axis")

    def sources(self):
        return [self.resolve_source("swiglu_backward_group_quant_with_dual_axis.cpp")]

    def schema(self):
        return (
            "swiglu_backward_group_quant_with_dual_axis("
            "Tensor grad_y, Tensor x, *, Tensor? weight=None, Tensor? y_origin=None, "
            "Tensor? group_index=None, float clamp_limit=-1.0, float alpha=1.0, "
            "float bias=0.0, int quant_mode=1, int dst_type=36) -> Tensor[]"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            grad_y,
            x,
            *,
            weight=None,
            y_origin=None,
            group_index=None,
            clamp_limit=-1.0,
            alpha=1.0,
            bias=0.0,
            quant_mode=1,
            dst_type=36,
        ):
            if quant_mode != 1:
                raise ValueError("quant_mode must be 1")
            if dst_type not in (FP8_E5M2, FP8_E4M3FN):
                raise ValueError("dst_type must be 35 or 36")
            if (weight is None) != (y_origin is None):
                raise ValueError("weight and y_origin must be provided together")
            if group_index is None and weight is not None:
                raise ValueError("weight and y_origin require group_index")

            width = x.shape[-1]
            rows = x.numel() // width
            fp8_dtype = (
                torch.float8_e5m2 if dst_type == FP8_E5M2 else torch.float8_e4m3fn
            )
            y1 = x.new_empty(x.shape, dtype=fp8_dtype)
            scale1 = x.new_empty(
                (*x.shape[:-1], (width + 63) // 64, 2), dtype=torch.float8_e8m0fnu
            )
            y2 = x.new_empty(x.shape, dtype=fp8_dtype)
            if group_index is None:
                scale2_shape = (*x.shape[:-2], (x.shape[-2] + 63) // 64, width, 2)
            else:
                scale2_shape = (rows // 64 + group_index.numel(), width, 2)
            scale2 = x.new_empty(scale2_shape, dtype=torch.float8_e8m0fnu)
            outputs = [y1, scale1, y2, scale2]
            if weight is not None:
                outputs.append(x.new_empty(weight.shape, dtype=weight.dtype))
            return outputs


builder = SwigluBackwardGroupQuantWithDualAxisOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _swiglu_backward_group_quant_with_dual_axis(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    y_origin: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    clamp_limit: float = -1.0,
    alpha: float = 1.0,
    bias: float = 0.0,
    quant_mode: int = 1,
    dst_type: int = 36,
) -> List[torch.Tensor]:
    module = builder.load()
    return module.swiglu_backward_group_quant_with_dual_axis(
        grad_y,
        x,
        weight,
        y_origin,
        group_index,
        clamp_limit,
        alpha,
        bias,
        quant_mode,
        dst_type,
    )


def swiglu_backward_group_quant_with_dual_axis(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    y_origin: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    clamp_limit: float = -1.0,
    alpha: float = 1.0,
    bias: float = 0.0,
    quant_mode: int = 1,
    dst_type: int = 36,
) -> List[torch.Tensor]:
    return torch.ops.cann_ops_nn.swiglu_backward_group_quant_with_dual_axis(
        grad_y,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=clamp_limit,
        alpha=alpha,
        bias=bias,
        quant_mode=quant_mode,
        dst_type=dst_type,
    )
