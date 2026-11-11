#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Ascend 950 references for ordinary and grouped Torch RotateQuant APIs."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from rotate_quant_kernel_golden import (
    _as_numpy,
    compare_grouped_outputs,
    compute_grouped_rotate_mx,
    compute_rotate_mx,
    np_fp4_e2m1,
    np_mx_scale,
)

_DST_TO_MX = {
    296: "float4_e2m1",
    23: "float8_e5m2",
    24: "float8_e4m3fn",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
}


def _mx_dtype(dst_dtype):
    # Direct callers may use torch.dtype; CSV/schema callers use integer IDs.
    name = str(dst_dtype).split(".")[-1]
    if name in ("float8_e5m2", "float8_e4m3fn"):
        return name
    if dst_dtype not in _DST_TO_MX:
        raise ValueError("Ascend 950 E2E golden requires an explicit MX dst_dtype")
    return _DST_TO_MX[dst_dtype]


def _unpack_fp4(value):
    """Decode the public uint8 FP4 carrier without numerically converting its bytes."""
    packed = _as_numpy(value)
    if packed.dtype != np.dtype(np.uint8):
        raise ValueError("Packed FP4 output dtype must equal uint8")
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes.view(np_fp4_e2m1)


def _logical_output(value, golden, name):
    """Validate public Torch dtypes before using byte views for NumPy comparison."""
    if isinstance(value, torch.Tensor):
        dtype_names = {
            "float4_e2m1": "uint8",
            "float8_e4m3fn": "float8_e4m3fn",
            "float8_e5m2": "float8_e5m2",
            "float8_e8m0": "float8_e8m0fnu",
        }
        expected = getattr(torch, dtype_names[golden.dtype.name])
        if value.dtype != expected:
            return None, {
                "pass": False,
                "precision": 0.0,
                "error_info": f"{name} dtype mismatch: actual={value.dtype}, expected={expected}",
            }
        if golden.dtype == np.dtype(np_fp4_e2m1):
            return _unpack_fp4(value), None
    array = _as_numpy(value)
    if golden.dtype == np.dtype(np_fp4_e2m1) and array.dtype == np.dtype(np.uint8):
        return _unpack_fp4(array), None
    if array.dtype != golden.dtype:
        return None, {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{name} dtype mismatch: actual={array.dtype}, expected={golden.dtype}",
        }
    return array, None


def _ordinary_torch_golden(
    x,
    rotation,
    *,
    alpha=None,
    dst_dtype=None,
    axis=-1,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    transpose_y=False,
):
    if (axis is not None and axis not in (-1, x.ndim - 1)) or transpose_y:
        raise ValueError("RotateQuant requires the last axis and transpose_y=False")
    dtype = _mx_dtype(dst_dtype)
    y, scale = compute_rotate_mx(
        x,
        rotation,
        alpha=alpha,
        mx_ele_dtype=dtype,
        round_mode="rint" if round_mode is None else str(round_mode),
        scale_alg=0 if scale_alg is None else int(scale_alg),
        dst_type_max=0.0 if dst_type_max is None else float(dst_type_max),
    )
    return [y, scale]


def _compare_torch_outputs(
    y,
    scale,
    golden_y,
    golden_scale,
    *,
    compare_context=None,
):
    golden_y, golden_scale = _as_numpy(golden_y), _as_numpy(golden_scale)
    if golden_y.dtype.name not in ("float4_e2m1", "float8_e4m3fn", "float8_e5m2"):
        raise ValueError("Torch golden y must retain its logical FP4/FP8 dtype")
    if golden_scale.dtype != np.dtype(np_mx_scale):
        raise ValueError("Torch golden scale must retain its E8M0 dtype")
    y, y_failure = _logical_output(y, golden_y, "y")
    scale, scale_failure = _logical_output(scale, golden_scale, "scale")
    results = compare_grouped_outputs(
        golden_y if y_failure is not None else y,
        golden_scale if scale_failure is not None else scale,
        golden_y,
        golden_scale,
        compare_context,
        preserve_tail=False,
    )
    for index, failure in enumerate((y_failure, scale_failure)):
        if failure is not None:
            results[index] = failure
    return results


class CannOpsNnRotateQuantTestSpec:
    """Packed uint8 FP4, native FP8 and E8M0 scales with logical NumPy references."""

    @staticmethod
    def golden(
        x,
        rotation,
        *,
        alpha=None,
        dst_dtype=None,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        transpose_y=False,
        group_list=None,
        group_list_type=0,
        **kwargs,
    ):
        if group_list_type not in (0, 1):
            raise ValueError("group_list_type must be 0 or 1")
        if group_list is not None:
            if (axis is not None and axis not in (-1, x.ndim - 1)) or transpose_y:
                raise ValueError(
                    "RotateQuant requires the last axis and transpose_y=False"
                )
            dtype = _mx_dtype(dst_dtype)
            y, scale = compute_grouped_rotate_mx(
                x,
                rotation,
                alpha,
                group_list,
                group_list_type,
                dtype,
                "rint" if round_mode is None else str(round_mode),
                0 if scale_alg is None else int(scale_alg),
                0.0 if dst_type_max is None else float(dst_type_max),
            )
            return [y, scale]
        y, scale = _ordinary_torch_golden(
            x,
            rotation,
            alpha=alpha,
            dst_dtype=dst_dtype,
            axis=axis,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type_max=dst_type_max,
            transpose_y=transpose_y,
        )
        return [y, scale]

    @staticmethod
    def compare(y, scale, golden_y, golden_scale, *, compare_context=None, **kwargs):
        return _compare_torch_outputs(
            y,
            scale,
            golden_y,
            golden_scale,
            compare_context=compare_context,
        )


__spec__ = {
    "torch.ops.cann_ops_nn.rotate_quant": "CannOpsNnRotateQuantTestSpec",
}
