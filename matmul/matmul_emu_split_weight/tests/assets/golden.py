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
import numpy as np
from ml_dtypes import bfloat16

__golden__ = {"kernel": {"matmul_emu_split_weight": "matmul_emu_split_weight_golden"}}


def customize_inputs(
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
    return x, w_high, w_low


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def matmul_emu_split_weight_golden(
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
    x, w_high, w_low = customize_inputs(
        x,
        w_high,
        w_low,
        transpose_x=transpose_x,
        transpose_w=transpose_w,
        scale=scale,
        y_dtype=y_dtype,
        **kwargs,
    )

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

    if x.dtype == bfloat16:
        out = out.astype(bfloat16).astype(np.float32)

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        out = _cast_output_dtype(out, output_dtypes[0])
    else:
        out = out.astype(np.float32)

    return [out]


class MatmulEmuSplitWeightAssets:
    golden = matmul_emu_split_weight_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    class ThirdPartyImpl:
        def __init__(
            self,
            *,
            transpose_x=False,
            transpose_w=False,
            scale=0.00390625,
            y_dtype=0,
            **kwargs,
        ):
            self.transpose_x = transpose_x
            self.transpose_w = transpose_w
            self.scale = scale

        def __call__(self, x, w_high, w_low, **kwargs):
            import torch

            x_f = x.to(torch.float32)
            w_high_f = w_high.to(torch.float32)
            w_low_f = w_low.to(torch.float32)

            if self.transpose_x:
                x_f = x_f.transpose(-2, -1)
            if self.transpose_w:
                w_high_f = w_high_f.transpose(-2, -1)
                w_low_f = w_low_f.transpose(-2, -1)

            out_high = torch.matmul(x_f, w_high_f)
            out_low = torch.matmul(x_f, w_low_f)

            out = out_high + out_low * self.scale
            return [out]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 5.0,
            "rmse_rtol": 2.0,
            "small_value": 1e-6,
            "small_value_atol": 1e-9,
        },
    }


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": bfloat16,
    }
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)
