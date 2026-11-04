# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch registration for the forward-only SwigluMxQuantWithDualAxis operator."""

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

BLOCK_SIZE = 32
ROW_BLOCK_SIZE = 64
MXSCALE_TAIL_DIM = 2
FP4_PER_BYTE = 2
DST_TYPE_FP8_E5M2 = 35
DST_TYPE_FP8_E4M3FN = 36
DST_TYPE_FP4_E2M1 = 40
DST_TYPE_FP4_E1M2 = 41

_FP4_DST_TYPES = (DST_TYPE_FP4_E2M1, DST_TYPE_FP4_E1M2)
_OUTPUT_DTYPES = {
    DST_TYPE_FP8_E5M2: torch.float8_e5m2,
    DST_TYPE_FP8_E4M3FN: torch.float8_e4m3fn,
    DST_TYPE_FP4_E2M1: torch.uint8,  # FP4 packed as uint8
    DST_TYPE_FP4_E1M2: torch.uint8,
}


def _check_arguments(x, group_index, round_mode, scale_alg, dst_type):
    torch._check(
        not x.requires_grad, lambda: "swiglu_mx_quant_with_dual_axis is forward-only"
    )
    torch._check(x.dim() == 2, lambda: f"x must be 2-dimensional, but got {x.dim()}")
    torch._check(
        x.shape[1] % FP4_PER_BYTE == 0,
        lambda: f"x last dim must be even, but got {x.shape[1]}",
    )
    torch._check(
        x.dtype in (torch.float16, torch.bfloat16),
        lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
    )
    torch._check(
        dst_type in _OUTPUT_DTYPES,
        lambda: (
            "dst_type must be 35(FP8_E5M2), 36(FP8_E4M3FN), 40(FP4_E2M1) or 41(FP4_E1M2), "
            f"but got {dst_type}"
        ),
    )
    torch._check(
        scale_alg in (0, 1),
        lambda: f"scale_alg must be 0 (OCP) or 1 (cuBLAS, FP8 only), got {scale_alg}",
    )
    if dst_type in _FP4_DST_TYPES:
        torch._check(
            scale_alg == 0,
            lambda: f"scale_alg must be 0 (OCP) for FP4 dst_type, got {scale_alg}",
        )
        torch._check(
            round_mode in ("rint", "floor", "round"),
            lambda: f"round_mode must be rint, floor or round for FP4 dst_type, got {round_mode}",
        )
    else:
        torch._check(
            round_mode == "rint",
            lambda: f"round_mode must be rint for FP8 dst_type, got {round_mode}",
        )
    if group_index is not None:
        torch._check(
            group_index.dim() == 1,
            lambda: f"group_index must be 1-dimensional, got {group_index.dim()}",
        )
        torch._check(
            group_index.dtype in (torch.int32, torch.int64),
            lambda: f"group_index dtype must be int32 or int64, got {group_index.dtype}",
        )


def _output_shapes(rows, cols, groups, dst_type):
    """y1/y2 and the two MX scale shapes (see the op README shape contract)."""
    y = [rows, cols // FP4_PER_BYTE if dst_type in _FP4_DST_TYPES else cols]
    col_blocks = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale1 = [rows, (col_blocks + 1) // FP4_PER_BYTE, MXSCALE_TAIL_DIM]
    # the grouped -2 axis layout keeps one scale pair gap per group
    scale2_rows = (
        rows // ROW_BLOCK_SIZE + groups
        if groups
        else (rows + ROW_BLOCK_SIZE - 1) // ROW_BLOCK_SIZE
    )
    scale2 = [scale2_rows, cols, MXSCALE_TAIL_DIM]
    return y, scale1, scale2


class SwigluMxQuantWithDualAxisOpBuilder(OpBuilder):
    """SwigluMxQuantWithDualAxis builder (aclnnSwigluMxQuantWithDualAxis).

    Computes SwiGLU and then quantizes the result along both the -1 and -2 axes
    with block-wise MX dynamic quantization, emitting two low-precision tensors
    and their shared-exponent scales.
    """

    def __init__(self):
        super().__init__("swiglu_mx_quant_with_dual_axis")

    def sources(self):
        return [self.resolve_source("swiglu_mx_quant_with_dual_axis.cpp")]

    def schema(self) -> str:
        return (
            "swiglu_mx_quant_with_dual_axis(Tensor x, Tensor? group_index=None, *, "
            'bool activate_left=True, str round_mode="rint", int scale_alg=0, '
            "int dst_type=40, float max_dtype_value=0.0) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def swiglu_mx_quant_with_dual_axis_meta(
            x,
            group_index=None,
            *,
            activate_left=True,
            round_mode="rint",
            scale_alg=0,
            dst_type=DST_TYPE_FP4_E2M1,
            max_dtype_value=0.0,
        ):
            _check_arguments(x, group_index, round_mode, scale_alg, dst_type)
            y_shape, scale1_shape, scale2_shape = _output_shapes(
                x.shape[0],
                x.shape[1] // FP4_PER_BYTE,
                group_index.numel() if group_index is not None else 0,
                dst_type,
            )
            y_dtype = _OUTPUT_DTYPES[dst_type]
            scale_dtype = torch.float8_e8m0fnu
            return (
                x.new_empty(y_shape, dtype=y_dtype),
                x.new_empty(scale1_shape, dtype=scale_dtype),
                x.new_empty(y_shape, dtype=y_dtype),
                x.new_empty(scale2_shape, dtype=scale_dtype),
            )


builder = SwigluMxQuantWithDualAxisOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _swiglu_mx_quant_with_dual_axis_npu(
    x,
    group_index=None,
    *,
    activate_left=True,
    round_mode="rint",
    scale_alg=0,
    dst_type=DST_TYPE_FP4_E2M1,
    max_dtype_value=0.0,
):
    outputs = builder.load().swiglu_mx_quant_with_dual_axis(
        x,
        group_index,
        activate_left,
        round_mode,
        scale_alg,
        dst_type,
        max_dtype_value,
    )
    return tuple(output.detach() for output in outputs)


def swiglu_mx_quant_with_dual_axis(
    x,
    group_index=None,
    *,
    activate_left=True,
    round_mode="rint",
    scale_alg=0,
    dst_type=DST_TYPE_FP4_E2M1,
    max_dtype_value=0.0,
):
    """Fused SwiGLU followed by dual-axis (-1 and -2) MX block quantization."""
    return torch.ops.cann_ops_nn.swiglu_mx_quant_with_dual_axis.default(
        x,
        group_index,
        activate_left=activate_left,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type=dst_type,
        max_dtype_value=max_dtype_value,
    )
