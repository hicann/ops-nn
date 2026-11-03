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
SUPPORTED_ROUND_MODES = ("rint", "trunc")
DEFAULT_SCALE_ALG = 0
SCALE_ALG_CHOICES = (0, 1)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _check_x(x):
    if x.dim() < 2 or x.dim() > 7:
        raise RuntimeError(f"x rank should be in [2, 7], but got {x.dim()}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"x dtype should be float16 or bfloat16, but got {x.dtype}")
    if x.size(-1) % HIF4_BLOCK_SIZE != 0 or x.size(-2) % HIF4_BLOCK_SIZE != 0:
        raise RuntimeError(
            "x last and second last dimensions should be multiples of 64"
        )


def _check_round_mode(round_mode):
    if round_mode not in SUPPORTED_ROUND_MODES:
        raise RuntimeError(
            f"round_mode should be one of {SUPPORTED_ROUND_MODES}, but got {round_mode}"
        )


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


def _scale1_shape(x):
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.size(x.dim() - 1), HIF4_BLOCK_SIZE)
    shape.append(1)
    return shape


def _scale2_shape(x):
    shape = list(x.shape)
    shape[-2] = _ceil_div(x.size(x.dim() - 2), HIF4_BLOCK_SIZE)
    shape.append(1)
    return shape


class DynamicBlockHiF4QuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("dynamic_block_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("dynamic_block_hi_f4_quant.cpp")]

    def schema(self):
        return (
            "dynamic_block_hi_f4_quant(Tensor x, *, str round_mode='rint', "
            "int scale_alg=0) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dynamic_block_hi_f4_quant_meta(
            x, *, round_mode="rint", scale_alg=DEFAULT_SCALE_ALG
        ):
            _check_x(x)
            _check_round_mode(round_mode)
            _validate_scale_alg(scale_alg)
            y = x.new_empty(_y_shape(x), dtype=torch.uint8)
            y_scale1 = x.new_empty(_scale1_shape(x), dtype=torch.float32)
            y_scale2 = x.new_empty(_scale2_shape(x), dtype=torch.float32)
            return y, y_scale1, y_scale2


builder = DynamicBlockHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def dynamic_block_hi_f4_quant(x, *, round_mode="rint", scale_alg=DEFAULT_SCALE_ALG):
    _check_x(x)
    _check_round_mode(round_mode)
    _validate_scale_alg(scale_alg)
    op_module = builder.load()
    return op_module.dynamic_block_hi_f4_quant(x, round_mode, scale_alg)
