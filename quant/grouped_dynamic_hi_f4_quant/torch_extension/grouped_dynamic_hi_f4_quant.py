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
X_DIM_NUM = 2
GROUP_INDEX_DTYPE = torch.int32
DEFAULT_SCALE_ALG = 0
SCALE_ALG_CHOICES = (0, 1)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _check_x(x):
    if x.dim() != X_DIM_NUM:
        raise RuntimeError(f"x rank should be {X_DIM_NUM}, but got {x.dim()}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"x dtype should be float16 or bfloat16, but got {x.dtype}")
    if x.size(-1) % HIF4_BLOCK_SIZE != 0:
        raise RuntimeError("x last dimension should be a multiple of 64")


def _check_group_index(group_index):
    if group_index.dim() != 1:
        raise RuntimeError(f"group_index rank should be 1, but got {group_index.dim()}")
    if group_index.numel() == 0:
        raise RuntimeError("group_index does not support empty tensor")
    if group_index.dtype != GROUP_INDEX_DTYPE:
        raise RuntimeError(
            f"group_index dtype should be int32, but got {group_index.dtype}"
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


def _scale_shape(x, group_index):
    # Match the reserved group layout used by the operator, not the sum of group block counts.
    return [x.size(0) // HIF4_BLOCK_SIZE + group_index.size(0), x.size(1), 1]


class GroupedDynamicHiF4QuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("grouped_dynamic_hi_f4_quant")

    def sources(self):
        return [self.resolve_source("grouped_dynamic_hi_f4_quant.cpp")]

    def schema(self):
        return "grouped_dynamic_hi_f4_quant(Tensor x, Tensor group_index, int scale_alg=0) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def grouped_dynamic_hi_f4_quant_meta(
            x, group_index, scale_alg=DEFAULT_SCALE_ALG
        ):
            _check_x(x)
            _check_group_index(group_index)
            if x.device != group_index.device:
                raise RuntimeError("x and group_index should be on the same device")
            _validate_scale_alg(scale_alg)
            y = x.new_empty(_y_shape(x), dtype=torch.uint8)
            y_scale = x.new_empty(_scale_shape(x, group_index), dtype=torch.float32)
            return y, y_scale


builder = GroupedDynamicHiF4QuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def grouped_dynamic_hi_f4_quant(x, group_index, scale_alg=DEFAULT_SCALE_ALG):
    _check_x(x)
    _check_group_index(group_index)
    if x.device != group_index.device:
        raise RuntimeError("x and group_index should be on the same device")
    _validate_scale_alg(scale_alg)
    op_module = builder.load()
    return op_module.grouped_dynamic_hi_f4_quant(x, group_index, scale_alg)
