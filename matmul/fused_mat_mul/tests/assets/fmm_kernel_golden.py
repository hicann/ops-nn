#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Kernel-level golden for FusedMatMul.

op: fused_mat_mul   formula: y = FUSED_OP(x1 @ x2 + bias, x3)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def _kernel_compute(
    x1,
    x2,
    bias=None,
    x3=None,
    *,
    transpose_x1=False,
    transpose_x2=False,
    fused_op_type="",
    enable_hf32=False,
    **kwargs,
):
    """Core fused_mat_mul numpy simulation.  x1/x2 already in ND format."""
    x1_dtype = x1.dtype

    if enable_hf32 and x1_dtype == np.float32:
        x1 = _util.hf32_truncate_np(x1)
        x2 = _util.hf32_truncate_np(x2)

    if x1_dtype in (np.float16, _util.np_bfloat16):
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)
        comp_dtype = np.float32
    else:
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        comp_dtype = np.float64

    if transpose_x1:
        x1 = np.swapaxes(x1, -2, -1)
    if transpose_x2:
        x2 = np.swapaxes(x2, -2, -1)

    mm_out = np.matmul(x1, x2)

    if x1_dtype == np.float32:
        mm_out = mm_out.astype(np.float32)

    if bias is not None and x1.shape[-1] != 0:
        bias = bias.astype(comp_dtype)
        mm_out = mm_out.astype(comp_dtype) + bias

    if fused_op_type in ("add", "mul"):
        if enable_hf32 and x3 is not None and x3.dtype == np.float32:
            x3 = _util.hf32_truncate_np(x3)
        output_dtypes = kwargs.get("output_dtypes", None)
        if output_dtypes is not None and x1_dtype in (np.float16, _util.np_bfloat16):
            mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0]).astype(
                comp_dtype
            )

    if fused_op_type == "relu":
        mm_out = np.maximum(mm_out, 0)
    elif fused_op_type == "add":
        x3 = x3.astype(comp_dtype)
        mm_out = mm_out + x3
    elif fused_op_type == "mul":
        x3 = x3.astype(comp_dtype)
        mm_out = mm_out * x3
    elif fused_op_type == "gelu_erf":
        mm_out = 0.5 * mm_out * (1.0 + np.vectorize(math.erf)(mm_out / np.sqrt(2.0)))
    elif fused_op_type == "gelu_tanh":
        mm_out = (
            0.5
            * mm_out
            * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (mm_out + 0.044715 * mm_out**3)))
        )

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


class FusedMatMulTestSpec:
    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x3=None,
        *,
        transpose_x1=False,
        transpose_x2=False,
        fused_op_type="",
        opImplMode=0,
        enable_hf32=False,
        **kwargs,
    ):
        """Kernel golden: NZ->ND conversion + core matmul + fused op."""
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())

        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            if ori_shape is not None and tuple(x2.shape) != tuple(ori_shape):
                x2 = _util.nz_to_nd(x2, ori_shape)

        return _kernel_compute(
            x1,
            x2,
            bias,
            x3,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            fused_op_type=fused_op_type,
            enable_hf32=enable_hf32,
            **kwargs,
        )


__spec__ = {
    "fused_mat_mul": "FusedMatMulTestSpec",
}
