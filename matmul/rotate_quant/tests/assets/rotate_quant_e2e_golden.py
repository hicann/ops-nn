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

sys.path.insert(0, os.path.dirname(__file__))
from rotate_quant_kernel_golden import (
    _as_numpy,
    _shape_dtype_failure,
    compare_grouped_outputs,
    compare_quantized,
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
    if name == "float4_e2m1fn_x2":
        return "float4_e2m1"
    if dst_dtype not in _DST_TO_MX:
        raise ValueError("Ascend 950 E2E golden requires an explicit MX dst_dtype")
    return _DST_TO_MX[dst_dtype]


def _pack_fp4(values):
    """Two E2M1 codes per byte: the first logical element is the low nibble."""
    codes = np.ascontiguousarray(values).view(np.uint8) & 0x0F
    if codes.shape[-1] % 2:
        raise ValueError("Packed FP4 requires an even last dimension")
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _unpack_fp4(packed):
    packed = _as_numpy(packed)
    if packed.dtype != np.uint8:
        raise ValueError("Torch FP4 storage must be uint8")
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes.view(np_fp4_e2m1)


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
    return [_pack_fp4(y) if dtype == "float4_e2m1" else y, scale]


def _compare_torch_outputs(
    y,
    scale,
    golden_y,
    golden_scale,
    *,
    compare_context=None,
):
    y, golden_y = _as_numpy(y), _as_numpy(golden_y)
    y_failure = _shape_dtype_failure(y, golden_y)
    if y_failure is None and golden_y.dtype == np.uint8:
        y, golden_y = _unpack_fp4(y), _unpack_fp4(golden_y)

    scale, golden_scale = _as_numpy(scale), _as_numpy(golden_scale)
    valid = scale.dtype == np.uint8 and golden_scale.dtype == np.uint8
    if valid:
        return compare_grouped_outputs(
            y,
            scale.view(np_mx_scale),
            golden_y,
            golden_scale.view(np_mx_scale),
            compare_context,
            preserve_tail=False,
        )
    return [
        y_failure or compare_quantized(y, golden_y, compare_context),
        {
            "pass": False,
            "precision": 0.0,
            "error_info": "Torch scale must retain its uint8 E8M0 storage dtype",
        },
    ]


class CannOpsNnRotateQuantTestSpec:
    """Operator-repository interface: packed FP4 Y and raw uint8 scale."""

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
            return [
                _pack_fp4(y) if dtype == "float4_e2m1" else y,
                scale.view(np.uint8).copy(),
            ]
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
        # Torch exports raw E8M0 codes, including the zero padding code.
        return [y, scale.view(np.uint8).copy()]

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
