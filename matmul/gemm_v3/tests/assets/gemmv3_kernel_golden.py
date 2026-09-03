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
"""Kernel-level golden for GemmV3.

op: gemm_v3   formula: y = alpha * a @ b + beta * c
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def _kernel_compute(
    a,
    b,
    c=None,
    *,
    transpose_a=False,
    transpose_b=False,
    alpha=1.0,
    beta=1.0,
    enable_hf32=False,
    **kwargs,
):
    a_dtype = a.dtype

    if enable_hf32 and a_dtype == np.float32:
        a = _util.hf32_truncate_np(a)
        b = _util.hf32_truncate_np(b)

    if a_dtype in (np.float16, _util.np_bfloat16):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        comp_dtype = np.float32
    else:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        comp_dtype = np.float64

    if transpose_a:
        a = np.swapaxes(a, -2, -1)
    if transpose_b:
        b = np.swapaxes(b, -2, -1)

    mm_out = np.matmul(a, b)
    mm_out = mm_out * alpha

    # If beta is 0, C is ignored and nan/inf in it must not propagate.
    if c is not None and beta != 0:
        c = c.astype(comp_dtype)
        mm_out = mm_out + beta * c

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


class GemmV3TestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        a,
        b,
        c=None,
        *,
        transpose_a=False,
        transpose_b=False,
        alpha=1.0,
        beta=1.0,
        enable_hf32=False,
        **kwargs,
    ):
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())

        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            if ori_shape is not None and tuple(b.shape) != tuple(ori_shape):
                b = _util.nz_to_nd(b, ori_shape)

        return _kernel_compute(
            a,
            b,
            c,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            enable_hf32=enable_hf32,
            **kwargs,
        )


__spec__ = {
    "gemm_v3": "GemmV3TestSpec",
}
