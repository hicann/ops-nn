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

HIF4_BLOCK_SIZE = 64
FP4_IN_BYTE = 2
DEFAULT_SCALE_ALG = 0
SCALE_ALG_CHOICES = (0, 1)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _check_x(x):
    if x.dim() < 2 or x.dim() > 7:
        raise RuntimeError(f"x rank should be in [2, 7], but got {x.dim()}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"x dtype should be float16 or bfloat16, but got {x.dtype}")
    if x.size(-1) % HIF4_BLOCK_SIZE != 0:
        raise RuntimeError("x last dimension should be a multiple of 64")


def _check_axis(x, axis):
    if axis < -x.dim() or axis >= x.dim():
        raise RuntimeError(
            f"axis is out of input dimension range [{-x.dim()}, {x.dim()}), but got {axis}"
        )
    axis_idx = axis % x.dim()
    if axis_idx not in (x.dim() - 1, x.dim() - 2):
        raise RuntimeError("axis should select the last or second last dimension")
    if axis_idx == x.dim() - 2 and x.size(-2) % HIF4_BLOCK_SIZE != 0:
        raise RuntimeError("x second last dimension should be a multiple of 64")


def _validate_scale_alg(scale_alg):
    """scale_alg: 0-标准阈值(4/2), 1-放宽阈值(3.75/1.875)"""
    if scale_alg not in SCALE_ALG_CHOICES:
        raise RuntimeError(
            f"scale_alg must be one of {SCALE_ALG_CHOICES}, but got {scale_alg}"
        )


def _y_shape(x):
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.size(x.dim() - 1), FP4_IN_BYTE)
    return shape


def _scale_shape(x, axis):
    shape = list(x.shape)
    shape[axis] = _ceil_div(x.size(axis), HIF4_BLOCK_SIZE)
    return shape + [1]


class DynamicHiF4QuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("dynamic_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("dynamic_hi_f4_quant.cpp")]

    def schema(self):
        return "dynamic_hi_f4_quant(Tensor x, *, int axis=-1, int scale_alg=0) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dynamic_hi_f4_quant_meta(x, *, axis=-1, scale_alg=DEFAULT_SCALE_ALG):
            _check_x(x)
            _check_axis(x, axis)
            _validate_scale_alg(scale_alg)
            y = x.new_empty(_y_shape(x), dtype=torch.uint8)
            y_scale = x.new_empty(_scale_shape(x, axis), dtype=torch.float32)
            return y, y_scale


builder = DynamicHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def dynamic_hi_f4_quant(x, *, axis=-1, scale_alg=DEFAULT_SCALE_ALG):
    _check_x(x)
    _check_axis(x, axis)
    _validate_scale_alg(scale_alg)
    op_module = builder.load()
    return op_module.dynamic_hi_f4_quant(x, axis, scale_alg)
