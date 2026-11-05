# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import List, Optional

import torch
import torch_npu  # noqa: F401
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

from ._perm_defaults import _DEFAULT_PERM_X1, _DEFAULT_PERM_X2, _DEFAULT_PERM_Y

_SUPPORTED_PERM_X1 = ((0, 1, 2), (1, 0, 2))
_SUPPORTED_PERM_X2 = ((0, 1, 2), (0, 2, 1))
_SUPPORTED_PERM_Y = ((1, 0, 2),)
_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SCALE_DTYPES = tuple(
    dtype
    for dtype in (torch.int64, getattr(torch, "uint64", None))
    if isinstance(dtype, torch.dtype)
)


def _resolve_perm(
    perm: Optional[List[int]], default: List[int], name: str
) -> List[int]:
    result = [int(value) for value in perm] if perm is not None else list(default)
    torch._check(len(result) == 3, lambda: f"{name} must have exactly 3 elements")
    return result


def _check_inputs(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    perm_x1: List[int],
    perm_x2: List[int],
    perm_y: List[int],
) -> None:
    torch._check(x1.dim() == 3, lambda: f"x1 must be 3-dimensional, but got {x1.dim()}")
    torch._check(x2.dim() == 3, lambda: f"x2 must be 3-dimensional, but got {x2.dim()}")
    torch._check(
        x1.dtype == x2.dtype,
        lambda: f"x1 and x2 must have the same dtype, but got {x1.dtype} and {x2.dtype}",
    )
    torch._check(
        x1.dtype in _SUPPORTED_INPUT_DTYPES,
        lambda: f"x1/x2 dtype must be float16, bfloat16 or float32, but got {x1.dtype}",
    )
    torch._check(
        tuple(perm_x1) in _SUPPORTED_PERM_X1,
        lambda: f"perm_x1 only supports {[list(p) for p in _SUPPORTED_PERM_X1]}, but got {perm_x1}",
    )
    torch._check(
        tuple(perm_x2) in _SUPPORTED_PERM_X2,
        lambda: f"perm_x2 only supports {[list(p) for p in _SUPPORTED_PERM_X2]}, but got {perm_x2}",
    )
    torch._check(
        tuple(perm_y) in _SUPPORTED_PERM_Y,
        lambda: f"perm_y only supports {[list(p) for p in _SUPPORTED_PERM_Y]}, but got {perm_y}",
    )
    torch._check(
        x1.size(perm_x1[0]) == x2.size(perm_x2[0]),
        lambda: "x1 and x2 batch dimensions must match",
    )
    torch._check(
        x1.size(perm_x1[2]) == x2.size(perm_x2[1]),
        lambda: "x1 and x2 K dimensions must match",
    )
    if bias is not None:
        torch._check(
            bias.dtype in _SUPPORTED_INPUT_DTYPES,
            lambda: f"bias dtype must be float16, bfloat16 or float32, but got {bias.dtype}",
        )
    if scale is not None:
        torch._check(
            scale.dtype in _SCALE_DTYPES,
            lambda: f"scale dtype must be int64 or uint64, but got {scale.dtype}",
        )
        torch._check(
            scale.dim() == 1,
            lambda: f"scale must be 1-dimensional, but got {scale.dim()}",
        )
        torch._check(
            x1.dtype == torch.float16,
            lambda: "scale is only supported for float16 input with int8 output, "
            f"but got x1 dtype {x1.dtype}",
        )


def _infer_output_shape(
    x1: torch.Tensor,
    x2: torch.Tensor,
    perm_x1: List[int],
    perm_x2: List[int],
    has_scale: bool,
    batch_split_factor: int,
) -> List[int]:
    m_dim = x1.size(perm_x1[1])
    batch_dim = x1.size(perm_x1[0])
    n_dim = x2.size(perm_x2[2])
    if has_scale:
        return [m_dim, 1, batch_dim * n_dim]
    if batch_split_factor > 1:
        torch._check(
            batch_dim % batch_split_factor == 0,
            lambda: "batch_split_factor must be a factor of batch, "
            f"but got batch={batch_dim}, batch_split_factor={batch_split_factor}",
        )
        return [batch_split_factor, m_dim, batch_dim * n_dim // batch_split_factor]
    return [m_dim, batch_dim, n_dim]


class TransposeBatchMatMulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("transpose_batch_mat_mul")

    def sources(self) -> list:
        return [self.resolve_source("transpose_batch_mat_mul.cpp")]

    def schema(self) -> str:
        return (
            "transpose_batch_mat_mul("
            "Tensor x1, Tensor x2, *, Tensor? bias=None, Tensor? scale=None, "
            "int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None, "
            "bool enable_hf32=False, int? batch_split_factor=None"
            ") -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def transpose_batch_mat_mul_meta(
            x1: torch.Tensor,
            x2: torch.Tensor,
            *,
            bias: Optional[torch.Tensor] = None,
            scale: Optional[torch.Tensor] = None,
            perm_x1: Optional[List[int]] = None,
            perm_x2: Optional[List[int]] = None,
            perm_y: Optional[List[int]] = None,
            enable_hf32: bool = False,
            batch_split_factor: Optional[int] = None,
        ) -> torch.Tensor:
            perm_x1_real = _resolve_perm(perm_x1, _DEFAULT_PERM_X1, "perm_x1")
            perm_x2_real = _resolve_perm(perm_x2, _DEFAULT_PERM_X2, "perm_x2")
            perm_y_real = _resolve_perm(perm_y, _DEFAULT_PERM_Y, "perm_y")
            _check_inputs(x1, x2, bias, scale, perm_x1_real, perm_x2_real, perm_y_real)

            has_scale = scale is not None
            batch_split_factor_value = (
                1 if batch_split_factor is None else int(batch_split_factor)
            )
            torch._check(
                batch_split_factor_value > 0,
                lambda: f"batch_split_factor must be positive, but got {batch_split_factor_value}",
            )
            if has_scale:
                torch._check(
                    batch_split_factor_value == 1,
                    lambda: "batch_split_factor must be 1 when scale is provided",
                )
                torch._check(
                    scale.numel()
                    == x2.size(perm_x2_real[0]) * x2.size(perm_x2_real[2]),
                    lambda: "scale length must be batch * N, "
                    f"but got {scale.numel()} and batch*N={x2.size(perm_x2_real[0]) * x2.size(perm_x2_real[2])}",
                )

            output_shape = _infer_output_shape(
                x1,
                x2,
                perm_x1_real,
                perm_x2_real,
                has_scale,
                batch_split_factor_value,
            )
            output_dtype = torch.int8 if has_scale else x1.dtype
            return torch.empty(output_shape, dtype=output_dtype, device="meta")


transpose_batch_mat_mul_builder = TransposeBatchMatMulOpBuilder()
transpose_batch_mat_mul_builder._ensure_initialized()


@impl(get_as_library(), transpose_batch_mat_mul_builder.name, "PrivateUse1")
def transpose_batch_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    perm_x1: Optional[List[int]] = None,
    perm_x2: Optional[List[int]] = None,
    perm_y: Optional[List[int]] = None,
    enable_hf32: bool = False,
    batch_split_factor: Optional[int] = None,
) -> torch.Tensor:
    op_module = transpose_batch_mat_mul_builder.load()
    return op_module.transpose_batch_mat_mul(
        x1,
        x2,
        bias,
        scale,
        perm_x1,
        perm_x2,
        perm_y,
        enable_hf32,
        batch_split_factor,
    )
