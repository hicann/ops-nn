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

DST_TYPE_FLOAT16 = 1
DST_TYPE_BFLOAT16 = 27
FP4_IN_BYTE = 2
HIF4_BLOCK_SIZE = 64
X_DIM_NUM_MIN = 2
X_DIM_NUM_MAX = 7
SCALE_DIM_NUM_MIN = 3
SCALE_DIM_NUM_MAX = 8
SUPPORTED_DST_TYPES = (DST_TYPE_FLOAT16, DST_TYPE_BFLOAT16)


def _check_dst_type(dst_type):
    if dst_type not in SUPPORTED_DST_TYPES:
        raise RuntimeError(
            f"dst_type should be one of {SUPPORTED_DST_TYPES} "
            f"(1=float16, 27=bfloat16), but got {dst_type}"
        )


def _check_rank(x, scale):
    if x.dim() < X_DIM_NUM_MIN or x.dim() > X_DIM_NUM_MAX:
        raise RuntimeError(
            f"x rank should be in [{X_DIM_NUM_MIN}, {X_DIM_NUM_MAX}], but got {x.dim()}"
        )
    if scale.dim() < SCALE_DIM_NUM_MIN or scale.dim() > SCALE_DIM_NUM_MAX:
        raise RuntimeError(
            f"scale rank should be in [{SCALE_DIM_NUM_MIN}, {SCALE_DIM_NUM_MAX}], but got {scale.dim()}"
        )


def _check_inputs(x, scale):
    _check_rank(x, scale)
    if x.dtype != torch.uint8:
        raise RuntimeError(
            f"x dtype should be uint8 (packed HiFloat4), but got {x.dtype}"
        )
    if scale.dtype != torch.float32:
        raise RuntimeError(
            f"scale dtype should be float32 (packed HiFloat4_SCALE storage), but got {scale.dtype}"
        )
    if x.device != scale.device:
        raise RuntimeError("x and scale should be on the same device")
    if not scale.is_contiguous():
        raise RuntimeError("scale should be contiguous")
    if (
        x.size(-1) % (HIF4_BLOCK_SIZE // FP4_IN_BYTE) != 0
        or x.size(-2) % HIF4_BLOCK_SIZE != 0
    ):
        raise RuntimeError(
            "x logical last and second last dimensions should be multiples of 64"
        )


def _y_shape(x):
    shape = list(x.shape)
    shape[-1] = x.size(x.dim() - 1) * FP4_IN_BYTE
    return shape


def _y_dtype(dst_type):
    return torch.float16 if dst_type == DST_TYPE_FLOAT16 else torch.bfloat16


class AntiHiF4QuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("anti_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("anti_hi_f4_quant.cpp")]

    def schema(self):
        return "anti_hi_f4_quant(Tensor x, Tensor scale, *, int dst_type=27) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def anti_hi_f4_quant_meta(x, scale, *, dst_type=27):
            _check_dst_type(dst_type)
            _check_inputs(x, scale)
            y = x.new_empty(_y_shape(x), dtype=_y_dtype(dst_type))
            return y


builder = AntiHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def anti_hi_f4_quant(x, scale, *, dst_type=27):
    _check_dst_type(dst_type)
    _check_inputs(x, scale)
    op_module = builder.load()
    return op_module.anti_hi_f4_quant(x, scale, dst_type)
