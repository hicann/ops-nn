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

BLOCK_FP8_QUANT_MODE = 0
MX_QUANT_MODE = 1
STATIC_HIFP8_QUANT_MODE = 2
DYNAMIC_HIFP8_QUANT_MODE = 3
BLOCK_FP8_BLOCK_SIZE = 128
MX_BLOCK_SIZE = 32
MX_SCALE_ALIGN = 2
TORCH_FLOAT8_E5M2 = 23
TORCH_FLOAT8_E4M3FN = 24
FLOAT8_E5M2 = 291
FLOAT8_E4M3FN = 292
FP4_E2M1 = 296
FP4_E1M2 = 297


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _swiglu_shape(x):
    if x.dim() < 1:
        raise RuntimeError("x rank should be greater than 0")
    last_dim = x.size(x.dim() - 1)
    if last_dim % 2 != 0:
        raise RuntimeError("x last dim size should be even")
    shape = list(x.shape)
    shape[-1] = last_dim // 2
    return shape


def _is_fp4_dtype(dst_type):
    return dst_type in (FP4_E2M1, FP4_E1M2)


def _quant_output_dtype(dst_type, quant_mode):
    if quant_mode in (STATIC_HIFP8_QUANT_MODE, DYNAMIC_HIFP8_QUANT_MODE):
        return torch.uint8
    if dst_type in (TORCH_FLOAT8_E5M2, FLOAT8_E5M2):
        return torch.float8_e5m2
    if dst_type in (TORCH_FLOAT8_E4M3FN, FLOAT8_E4M3FN):
        return torch.float8_e4m3fn
    if _is_fp4_dtype(dst_type):
        if quant_mode != MX_QUANT_MODE:
            raise RuntimeError("dst_type FLOAT4_E2M1/FLOAT4_E1M2 requires quant_mode=1")
        return torch.uint8
    raise RuntimeError(f"unsupported dst_type: {dst_type}")


def _quant_output_shape(x, dst_type, quant_mode):
    shape = _swiglu_shape(x)
    if quant_mode == MX_QUANT_MODE and _is_fp4_dtype(dst_type):
        shape[-1] = _ceil_div(shape[-1], 2)
    return shape


def _scale_shape(x, group_index, quant_mode):
    swiglu_last_dim = x.size(x.dim() - 1) // 2
    if quant_mode == STATIC_HIFP8_QUANT_MODE:
        return [0]
    if quant_mode == DYNAMIC_HIFP8_QUANT_MODE:
        if group_index is not None:
            return list(group_index.shape)
        return [1]

    shape = list(x.shape[:-1])
    if quant_mode == MX_QUANT_MODE:
        tail_dim = _ceil_div(swiglu_last_dim, MX_BLOCK_SIZE)
        tail_dim = _ceil_div(tail_dim, MX_SCALE_ALIGN)
        shape.extend([tail_dim, MX_SCALE_ALIGN])
    else:
        shape.append(_ceil_div(swiglu_last_dim, BLOCK_FP8_BLOCK_SIZE))
    return shape


class SwigluGroupQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_group_quant")

    def sources(self):
        return ["csrc/activation/swiglu_group_quant.cpp"]

    def schema(self):
        return (
            "swiglu_group_quant(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, Tensor? scale=None, "
            "int dst_type=291, int quant_mode=0, int block_size=0, bool round_scale=False, "
            "float clamp_limit=-1.0, float dst_type_max=15.0, bool output_origin=False) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_group_quant_meta(
            x,
            *,
            weight=None,
            group_index=None,
            scale=None,
            dst_type=291,
            quant_mode=0,
            block_size=0,
            round_scale=False,
            clamp_limit=-1.0,
            dst_type_max=15.0,
            output_origin=False,
        ):
            y = x.new_empty(
                _quant_output_shape(x, dst_type, quant_mode),
                dtype=_quant_output_dtype(dst_type, quant_mode),
            )
            y_scale_dtype = (
                torch.float8_e8m0fnu if quant_mode == MX_QUANT_MODE else torch.float32
            )
            y_scale = x.new_empty(
                _scale_shape(x, group_index, quant_mode), dtype=y_scale_dtype
            )
            y_origin = x.new_empty(_swiglu_shape(x) if output_origin else [0])
            return y, y_scale, y_origin


builder = SwigluGroupQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def swiglu_group_quant(
    x,
    *,
    weight=None,
    group_index=None,
    scale=None,
    dst_type=291,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=-1.0,
    dst_type_max=15.0,
    output_origin=False,
):
    op_module = builder.load()
    return op_module.swiglu_group_quant(
        x,
        weight,
        group_index,
        scale,
        dst_type,
        quant_mode,
        block_size,
        round_scale,
        clamp_limit,
        dst_type_max,
        output_origin,
    )
