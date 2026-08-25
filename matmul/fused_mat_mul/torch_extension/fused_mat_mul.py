# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
import torch_npu
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


def _infer_output_shape(x: torch.Tensor, x2: torch.Tensor):
    torch._check(
        x.dim() >= 2, lambda: f"x must have at least 2 dimensions, but got {x.dim()}"
    )
    torch._check(
        x2.dim() >= 2, lambda: f"x2 must have at least 2 dimensions, but got {x2.dim()}"
    )
    torch._check(
        x.shape[-1] == x2.shape[-2], lambda: "x and x2 K dimensions must match"
    )
    batch_shape = torch.broadcast_shapes(x.shape[:-2], x2.shape[:-2])
    return (*batch_shape, x.shape[-2], x2.shape[-1])


def _check_fused_matmul_inputs(x, x2, x3, fused_op_type):
    torch._check(x.dtype == x2.dtype, lambda: "x and x2 must have the same dtype")
    if fused_op_type in ("add", "mul"):
        torch._check(x3 is not None, lambda: "x3 must be provided for add and mul")
    else:
        torch._check(x3 is None, lambda: "x3 is only supported for add and mul")


def _get_cube_math_type() -> int:
    cube_math_type = torch_npu.npu.matmul.cube_math_type
    if cube_math_type is not None:
        return int(cube_math_type)
    if torch.npu.matmul.allow_hf32:
        return int(torch_npu.npu.CubeMathType.USE_HF32)
    return int(torch_npu.npu.CubeMathType.KEEP_DTYPE)


class FusedMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("fused_matmul")

    def sources(self) -> list:
        return [self.resolve_source("fused_matmul.cpp")]

    def schema(self) -> str:
        return (
            "fused_matmul(Tensor x, Tensor x2, *, Tensor? bias=None, Tensor? x3=None, "
            'Scalar? alpha=None, Scalar? beta=None, str fused_op_type="") -> Tensor'
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def fused_matmul_meta(
            x,
            x2,
            *,
            bias=None,
            x3=None,
            alpha=None,
            beta=None,
            fused_op_type="",
        ):
            _check_fused_matmul_inputs(x, x2, x3, fused_op_type)
            output_dtype = torch.float32 if fused_op_type == "16cast32" else x.dtype
            return x.new_empty(_infer_output_shape(x, x2), dtype=output_dtype)


fused_matmul_builder = FusedMatmulOpBuilder()
fused_matmul_builder._ensure_initialized()


@impl(get_as_library(), fused_matmul_builder.name, "PrivateUse1")
def fused_matmul(
    x: torch.Tensor,
    x2: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    x3: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    fused_op_type: str = "",
) -> torch.Tensor:
    _check_fused_matmul_inputs(x, x2, x3, fused_op_type)
    op_module = fused_matmul_builder.load()
    cube_math_type = _get_cube_math_type()
    return op_module.fused_matmul(
        x, x2, bias, x3, alpha, beta, fused_op_type, cube_math_type
    )
