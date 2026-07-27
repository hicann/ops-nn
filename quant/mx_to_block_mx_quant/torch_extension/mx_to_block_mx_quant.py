#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class MxToBlockMxQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("mx_to_block_mx_quant")

    def sources(self) -> list:
        return ["csrc/quant/mx_to_block_mx_quant.cpp"]

    def schema(self) -> str:
        return (
            "mx_to_block_mx_quant("
            "Tensor x, "
            "Tensor mxscale, "
            "*, "
            "int dst_type=292, "
            "int x_type=296 "
            ") -> (Tensor, Tensor, Tensor) "
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mx_to_block_mx_quant_meta(
            x: torch.Tensor,
            mxscale: torch.Tensor,
            *,
            dst_type: int = 292,
            x_type: int = 296,
        ):
            torch._check(x.numel() > 0, lambda: "Input tensor x should not be empty.")
            torch._check(
                mxscale.numel() > 0, lambda: "Input tensor mxscale should not be empty."
            )
            torch._check(
                x.dim() == 2 or x.dim() == 3,
                lambda: f"Input x rank must be 2 or 3, got {x.dim()}",
            )
            torch._check(
                mxscale.dim() == x.dim() + 1,
                lambda: f"Input mxscale rank must be x rank + 1, "
                f"but got mxscale rank {mxscale.dim()} for x rank {x.dim()}",
            )

            rank = x.dim()
            y_shape = list(x.shape)

            _FLOAT4_X_TYPES = {296, 297}
            torch._check(
                x_type in _FLOAT4_X_TYPES,
                lambda: f"x_type must be one of {{296, 297}}, got {x_type}",
            )
            torch._check(
                x.dtype == torch.uint8,
                lambda: f"When x_type is float4, x dtype must be uint8, got {x.dtype}",
            )
            y_shape[rank - 1] = x.size(rank - 1) * 2

            output_dtype = TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[dst_type]
            y = torch.empty(y_shape, dtype=output_dtype, device="meta")
            scale1 = torch.empty(mxscale.shape, dtype=torch.uint8, device="meta")

            BLOCK_SIZE = 32
            last_dim = y_shape[rank - 1]
            second_last_dim = x.size(rank - 2)
            scale2_dim0 = ((second_last_dim + BLOCK_SIZE - 1) // BLOCK_SIZE + 1) // 2
            scale2_shape = []
            if rank == 3:
                scale2_shape.append(x.size(0))
            scale2_shape.extend([scale2_dim0, last_dim, 2])
            scale2 = torch.empty(scale2_shape, dtype=torch.uint8, device="meta")

            return y, scale1, scale2


mx_to_block_mx_quant_builder = MxToBlockMxQuantOpBuilder()
mx_to_block_mx_quant_builder._ensure_initialized()


@impl(get_as_library(), mx_to_block_mx_quant_builder.name, "PrivateUse1")
def mx_to_block_mx_quant(
    x: torch.Tensor, mxscale: torch.Tensor, *, dst_type: int = 292, x_type: int = 296
):
    op_module_quant = mx_to_block_mx_quant_builder.load()
    return op_module_quant.mx_to_block_mx_quant(x, mxscale, dst_type, x_type)
