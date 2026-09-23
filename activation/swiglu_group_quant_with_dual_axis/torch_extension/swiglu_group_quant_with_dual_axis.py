# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch registration for the forward-only SwigluGroupQuantWithDualAxis operator."""

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

FLOAT8_E5M2 = 291
FLOAT8_E4M3FN = 292
ACL_FLOAT8_E5M2 = 35
ACL_FLOAT8_E4M3FN = 36
MX_QUANT_MODE = 1


def _geometry(x):
    rank = x.dim()
    if rank != 2:
        raise RuntimeError("x must be a 2D tensor")
    return x.shape[0], x.shape[1] // 2


def _output_dtype(dst_type):
    if dst_type in (FLOAT8_E5M2, ACL_FLOAT8_E5M2):
        return torch.float8_e5m2
    if dst_type in (FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN):
        return torch.float8_e4m3fn
    raise RuntimeError("dst_type must select FLOAT8_E5M2 or FLOAT8_E4M3FN")


def _output_shapes(rows, cols, groups, has_group, output_origin, x_shape=None):
    if not has_group and x_shape is not None:
        y = list(x_shape[:-1]) + [cols]
        scale1 = y[:-1] + [(cols + 63) // 64, 2]
        scale2 = y[:-2] + [(y[-2] + 63) // 64, cols, 2]
        return y, scale1, y, scale2, y if output_origin else [0]
    y = [rows, cols]
    scale1 = [rows, (cols + 63) // 64, 2]
    # The non-group route follows SwigluMxQuantWithDualAxis: pair the
    # 32-row scales and expose ceil(rows / 64) rows.  The group count is only
    # part of the grouped route's public shape.
    scale2_rows = rows // 64 + groups if has_group else (rows + 63) // 64
    return y, scale1, y, [scale2_rows, cols, 2], y if output_origin else [0]


def _check_forward_only(x, weight):
    if x.requires_grad or (weight is not None and weight.requires_grad):
        raise RuntimeError(
            "swiglu_group_quant_with_dual_axis is forward-only and does not support autograd"
        )


class SwigluGroupQuantWithDualAxisOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_group_quant_with_dual_axis")

    def sources(self):
        return [self.resolve_source("swiglu_group_quant_with_dual_axis.cpp")]

    def schema(self):
        return (
            "swiglu_group_quant_with_dual_axis(Tensor x, Tensor? weight=None, Tensor? group_index=None, *, "
            "int dst_type=292, int quant_mode=1, float clamp_limit=-1.0, bool output_origin=False, "
            "float alpha=1.0, float bias=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_group_quant_with_dual_axis_meta(
            x,
            weight=None,
            group_index=None,
            *,
            dst_type=FLOAT8_E4M3FN,
            quant_mode=MX_QUANT_MODE,
            clamp_limit=-1.0,
            output_origin=False,
            alpha=1.0,
            bias=0.0,
        ):
            rows, cols = _geometry(x)
            output_dtype = _output_dtype(dst_type)
            _check_forward_only(x, weight)
            shapes = _output_shapes(
                rows,
                cols,
                group_index.numel() if group_index is not None else 1,
                group_index is not None,
                output_origin,
                x.shape,
            )
            return (
                x.new_empty(shapes[0], dtype=output_dtype),
                x.new_empty(shapes[1], dtype=torch.float8_e8m0fnu),
                x.new_empty(shapes[2], dtype=output_dtype),
                x.new_empty(shapes[3], dtype=torch.float8_e8m0fnu),
                x.new_empty(shapes[4]),
            )


builder = SwigluGroupQuantWithDualAxisOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _swiglu_group_quant_with_dual_axis_npu(
    x,
    weight=None,
    group_index=None,
    *,
    dst_type=FLOAT8_E4M3FN,
    quant_mode=MX_QUANT_MODE,
    clamp_limit=-1.0,
    output_origin=False,
    alpha=1.0,
    bias=0.0,
):
    _check_forward_only(x, weight)
    outputs = builder.load().swiglu_group_quant_with_dual_axis(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        clamp_limit,
        output_origin,
        alpha,
        bias,
    )
    return tuple(output.detach() for output in outputs)


def swiglu_group_quant_with_dual_axis(
    x,
    weight=None,
    group_index=None,
    *,
    dst_type=FLOAT8_E4M3FN,
    quant_mode=MX_QUANT_MODE,
    clamp_limit=-1.0,
    output_origin=False,
    alpha=1.0,
    bias=0.0,
):
    """Fused Clipped-SwiGLU followed by two-axis MX quantization."""
    _check_forward_only(x, weight)
    return torch.ops.cann_ops_nn.swiglu_group_quant_with_dual_axis.default(
        x,
        weight,
        group_index,
        dst_type=dst_type,
        quant_mode=quant_mode,
        clamp_limit=clamp_limit,
        output_origin=output_origin,
        alpha=alpha,
        bias=bias,
    )
