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
"""Kernel-level golden for BatchMatMulV3.

op: batch_mat_mul_v3   formula: y = x1 @ x2 (+ bias)

Kernel golden receives numpy.ndarray inputs/outputs.  The FRACTAL_NZ layout of
x2 (when input_formats[1] == 'FRACTAL_NZ') is converted back to logical ND via
``matmul_golden_util.nz_to_nd`` before the matmul.  HF32 execution is simulated
when ``opImplMode == 64`` (the kernel-side alias for enable_hf32).
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def _kernel_compute(
    x1,
    x2,
    bias=None,
    *,
    adj_x1=False,
    adj_x2=False,
    opImplMode=0,
    **kwargs,
):
    """Core batch_mat_mul_v3 numpy simulation.  x1 and x2 are already in ND format."""
    x1_dtype = x1.dtype

    if opImplMode == 64 and x1_dtype == np.float32:
        x1 = _util.hf32_truncate_np(x1)
        x2 = _util.hf32_truncate_np(x2)

    if x1_dtype in (np.float16, _util.np_bfloat16):
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)
        bias_comp_dtype = np.float32
    else:
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        bias_comp_dtype = np.float64

    if adj_x1:
        x1 = np.swapaxes(x1, -2, -1)
    if adj_x2:
        x2 = np.swapaxes(x2, -2, -1)

    out = np.matmul(x1, x2)

    if bias is not None:
        bias = bias.astype(bias_comp_dtype)
        out = out + bias

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        out = _util.cast_output_dtype(out, output_dtypes[0])

    return [out]


class BatchMatMulV3TestSpec:
    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        offset_w=None,
        *,
        adj_x1=False,
        adj_x2=False,
        offset_x=0,
        opImplMode=0,
        **kwargs,
    ):
        """Kernel golden: NZ→ND conversion + core matmul computation."""
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
            adj_x1=adj_x1,
            adj_x2=adj_x2,
            opImplMode=opImplMode,
            **kwargs,
        )


__spec__ = {
    "batch_mat_mul_v3": "BatchMatMulV3TestSpec",
}
