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
"""Kernel-level golden for MatmulEmuSplitWeight.

op: matmul_emu_split_weight   formula: y = x @ w_high + scale * (x @ w_low)
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def _kernel_compute(
    x,
    w_high,
    w_low,
    *,
    transpose_x=False,
    transpose_w=False,
    scale=0.00390625,
    **kwargs,
):
    """Core matmul_emu_split_weight numpy simulation."""

    x_f = x.astype(np.float32)
    w_high_f = w_high.astype(np.float32)
    w_low_f = w_low.astype(np.float32)

    if transpose_x:
        x_f = np.swapaxes(x_f, -2, -1)
    if transpose_w:
        w_high_f = np.swapaxes(w_high_f, -2, -1)
        w_low_f = np.swapaxes(w_low_f, -2, -1)

    out_high = np.matmul(x_f, w_high_f)
    out_low = np.matmul(x_f, w_low_f)

    out = out_high + out_low * np.float32(scale)

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        out = _util.cast_output_dtype(out, output_dtypes[0])
    else:
        out = out.astype(np.float32)

    return [out]


class MatmulEmuSplitWeightKernelTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x,
        w_high,
        w_low,
        *,
        transpose_x=False,
        transpose_w=False,
        scale=0.00390625,
        y_dtype=0,
        **kwargs,
    ):
        """Kernel golden: core matmul emu split weight computation."""
        return _kernel_compute(
            x,
            w_high,
            w_low,
            transpose_x=transpose_x,
            transpose_w=transpose_w,
            scale=scale,
            y_dtype=y_dtype,
            **kwargs,
        )


__spec__ = {
    "matmul_emu_split_weight": "MatmulEmuSplitWeightKernelTestSpec",
}
