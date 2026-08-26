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
"""Kernel-level golden for QuantBatchMatmulInplaceAdd.

op: quant_batch_matmul_inplace_add
formula:
  MX:       y[m,n] = SUM_j( (SUM_k(x1_slice * x2_slice)) * (scale1[m,j] * scale2[j,n]) ) + y[m,n]
  HiFloat8: y[m,n] = (SUM_k(x1[m,k] * x2[k,n])) * trunc19(scale1 * scale2) + y[m,n]
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
    ),
)
import matmul_golden_util as _util
from matmul_quant_util import (
    unpack_groupsize,
    scale_generate,
    ceil_div,
)


def _scale_e8m0_to_f32(scale):
    return scale.astype(np.float32)


def _compute_mx(x1, x2, x1_scale, x2_scale, transpose_x1, transpose_x2, group_size):
    gs_m, gs_n, gs_k = unpack_groupsize(group_size)

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x1s = _scale_e8m0_to_f32(x1_scale)
    x2s = _scale_e8m0_to_f32(x2_scale)

    if transpose_x1:
        x1 = np.swapaxes(x1, -2, -1)
        x1s = np.swapaxes(x1s, -2, -1)
        if x1s.ndim == 3:
            x1s = x1s.reshape(x1s.shape[0] * x1s.shape[1], x1s.shape[2])
        x1s = np.swapaxes(x1s, -2, -1)
    else:
        if x1s.ndim == 3:
            x1s = x1s.reshape(x1s.shape[0], x1s.shape[1] * x1s.shape[2])

    if transpose_x2:
        x2 = np.swapaxes(x2, -2, -1)
        if x2s.ndim == 3:
            x2s = x2s.reshape(x2s.shape[0], x2s.shape[1] * x2s.shape[2])
        x2s = np.swapaxes(x2s, -2, -1)
    else:
        x2s = np.swapaxes(x2s, -2, -1)
        if x2s.ndim == 3:
            x2s = x2s.reshape(x2s.shape[0] * x2s.shape[1], x2s.shape[2])

    k = x1.shape[-1]

    if ceil_div(k, gs_k) % 2 != 0:
        x1s = x1s[:, :-1]
        x2s = x2s[:-1, :]

    x1s_broadcast = np.repeat(x1s, gs_k, axis=-1)
    x2s_broadcast = np.repeat(x2s, gs_k, axis=-2)

    x1_pad_len = x1s_broadcast.shape[-1] - k
    x2_pad_len = x2s_broadcast.shape[-2] - k
    if x1_pad_len > 0:
        x1 = np.pad(
            x1,
            [(0, 0)] * (x1.ndim - 1) + [(0, x1_pad_len)],
            mode="constant",
            constant_values=0,
        )
    if x2_pad_len > 0:
        x2 = np.pad(
            x2,
            [(0, 0)] * (x2.ndim - 2) + [(0, x2_pad_len)] + [(0, 0)],
            mode="constant",
            constant_values=0,
        )

    x1 = x1 * x1s_broadcast
    x2 = x2 * x2s_broadcast

    return np.matmul(x1, x2)


def _compute_hif8_tt(x1, x2, x1_scale, x2_scale, transpose_x1, transpose_x2):
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)

    if transpose_x1:
        x1 = np.swapaxes(x1, -2, -1)
    if transpose_x2:
        x2 = np.swapaxes(x2, -2, -1)

    matmul_out = np.matmul(x1, x2)

    s1 = x1_scale.astype(np.float32).reshape(-1)[0]
    s2 = x2_scale.astype(np.float32).reshape(-1)[0]
    merged = np.array(s1 * s2, dtype=np.float32)
    merged = scale_generate(merged)

    return matmul_out * merged


def _kernel_compute(
    x1,
    x2,
    x2_scale,
    y,
    x1_scale=None,
    *,
    transpose_x1=False,
    transpose_x2=False,
    group_size=0,
    **kwargs,
):
    is_mx = "e8m0" in str(x2_scale.dtype)

    if is_mx:
        acc = _compute_mx(
            x1, x2, x1_scale, x2_scale, transpose_x1, transpose_x2, group_size
        )
    else:
        acc = _compute_hif8_tt(x1, x2, x1_scale, x2_scale, transpose_x1, transpose_x2)

    y_out = acc + y.astype(np.float32)

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        y_out = _util.cast_output_dtype(y_out, output_dtypes[0])
    else:
        y_out = y_out.astype(np.float32)

    return [y_out]


class QuantBatchMatmulInplaceAddTestSpec:
    compare = _util.isclose_compare

    @classmethod
    def golden(
        cls,
        x1,
        x2,
        x2_scale,
        y,
        x1_scale=None,
        *,
        transpose_x1=False,
        transpose_x2=False,
        group_size=0,
        **kwargs,
    ):
        return _kernel_compute(
            x1,
            x2,
            x2_scale,
            y,
            x1_scale,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            **kwargs,
        )


__spec__ = {
    "quant_batch_matmul_inplace_add": "QuantBatchMatmulInplaceAddTestSpec",
}
