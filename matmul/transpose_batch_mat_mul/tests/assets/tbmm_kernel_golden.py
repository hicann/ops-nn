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
"""Kernel-level golden for TransposeBatchMatMul.

op: transpose_batch_mat_mul   formula: y = perm_y(perm_x1(x1) @ perm_x2(x2))

Kernel golden receives numpy.ndarray inputs/outputs.  The FRACTAL_NZ layout of
x2 (when input_formats[1] == 'FRACTAL_NZ') is converted back to logical ND via
``matmul_golden_util.nz_to_nd`` before the matmul.  HF32 execution is simulated
when ``enable_hf32`` is True.  Scale (INT64/UINT64) is decoded via
``matmul_quant_util.u64_to_deq_scale`` before quantization.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
from matmul_quant_util import u64_to_deq_scale


def _kernel_compute(
    x1,
    x2,
    bias=None,
    scale=None,
    *,
    perm_x1=(0, 1, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    enable_hf32=False,
    batch_split_factor=1,
    **kwargs,
):
    """Core transpose_batch_mat_mul numpy simulation.  x1 and x2 are already in ND format."""
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

    x1_t = np.transpose(x1, axes=list(perm_x1))

    if tuple(perm_x2) == (0, 2, 1):
        x2_t = np.swapaxes(x2, -2, -1)
    else:
        x2_t = x2

    mm_out = np.matmul(x1_t, x2_t)

    if scale is not None:
        if scale.dtype in (np.int64, np.uint64):
            scale = u64_to_deq_scale(scale)
        mm_out = np.transpose(mm_out, axes=list(perm_y))
        M, B, N = mm_out.shape
        mm_out = mm_out.reshape(M, B * N)
        scale_f = scale.astype(comp_dtype).reshape(1, B * N)
        mm_out = mm_out * scale_f
        mm_out = np.sign(mm_out) * np.floor(np.abs(mm_out) + 0.5)
        mm_out = mm_out.clip(-128, 127).astype(np.int8)
        mm_out = mm_out.reshape(M, 1, B * N)
    elif batch_split_factor > 1:
        B, M, N = mm_out.shape
        inner_batch = B // batch_split_factor
        mm_out = mm_out.reshape(batch_split_factor, inner_batch, M, N)
        mm_out = mm_out.transpose(0, 2, 1, 3)
        mm_out = mm_out.reshape(batch_split_factor, M, inner_batch * N)
    else:
        mm_out = np.transpose(mm_out, axes=list(perm_y))

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


class TransposeBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        scale=None,
        *,
        perm_x1=(0, 1, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        enable_hf32=False,
        batch_split_factor=1,
        **kwargs,
    ):
        """Kernel golden: NZ->ND conversion + core matmul computation."""
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
            scale,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            enable_hf32=enable_hf32,
            batch_split_factor=batch_split_factor,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        scale=None,
        *,
        perm_x1=(0, 1, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        enable_hf32=False,
        batch_split_factor=1,
        **kwargs,
    ):
        """Input preprocessing: scale generation only. NZ->ND is handled in golden()."""
        if scale is not None and scale.dtype in (np.int64, np.uint64):
            scale_shape = scale.shape
            scale_orig_dtype = scale.dtype
            fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(
                np.float32
            )
            u32 = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
            u32 &= np.uint32(0xFFFFE000)
            new_scale = np.zeros(scale_shape, np.uint64)
            new_scale |= u32.astype(np.uint64)
            new_scale |= np.uint64(1 << 46)
            if scale_orig_dtype == np.int64:
                new_scale = new_scale.astype(np.int64)
            scale = new_scale

        return x1, x2, bias, scale


__spec__ = {
    "transpose_batch_mat_mul": "TransposeBatchMatMulTestSpec",
}
