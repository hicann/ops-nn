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
__input__ = {"kernel": {"quant_batch_matmul_v4": "quant_batch_matmul_v4_inputs"}}

import numpy as np
import warnings

UNSIGNED_ONLY_DTYPES = ("float8_e8m0",)


def _check_and_fix_unsigned_dtype_nan(tensor, input_index, input_ranges, testcase_name):
    if tensor is None:
        return tensor
    dtype_str = str(tensor.dtype)
    if not any(d in dtype_str for d in UNSIGNED_ONLY_DTYPES):
        return tensor
    fp32_check = tensor.astype(np.float32)
    nan_count = int(np.isnan(fp32_check).sum())
    if nan_count == 0:
        return tensor
    orig_range = None
    if input_ranges is not None:
        try:
            orig_range = (
                input_ranges[input_index] if input_index < len(input_ranges) else None
            )
        except (TypeError, IndexError):
            pass
    orig_low, orig_high = -10, 10
    if orig_range is not None and len(orig_range) >= 2:
        orig_low = orig_range[0] if orig_range[0] is not None else -10
        orig_high = orig_range[1] if orig_range[1] is not None else 10
    new_high = max(abs(float(orig_low)), abs(float(orig_high)), 1.0)
    new_low = max(new_high * 0.001, 1e-6)
    warnings.warn(
        f"[{testcase_name}] Input {input_index} dtype={dtype_str} contains {nan_count} NaN "
        f"values (original range ({orig_low}, {orig_high}) includes negatives). "
        f"Regenerating with positive range ({new_low}, {new_high})."
    )
    new_data = np.random.uniform(new_low, new_high, tensor.shape).astype(np.float32)
    return new_data.astype(tensor.dtype)


def quant_batch_matmul_v4_inputs(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    y_scale=None,
    x1_offset=None,
    x2_offset=None,
    y_offset=None,
    x2_table=None,
    dtype: int = -1,
    compute_type: int = -1,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    group_size: int = -1,
    **kwargs,
):
    testcase_name = kwargs.get("testcase_name", "unknown")
    input_ranges = kwargs.get("input_ranges", None)

    x1_scale = _check_and_fix_unsigned_dtype_nan(
        x1_scale, 3, input_ranges, testcase_name
    )
    x2_scale = _check_and_fix_unsigned_dtype_nan(
        x2_scale, 4, input_ranges, testcase_name
    )

    if y_scale is not None and y_scale.dtype.name == "uint64":
        y_scale = _generate_u64_scale(y_scale.shape)

    if x2_scale is not None and x2_scale.dtype.name == "uint64":
        x2_scale = _generate_u64_scale(x2_scale.shape)

    return (
        x1,
        x2,
        bias,
        x1_scale,
        x2_scale,
        y_scale,
        x1_offset,
        x2_offset,
        y_offset,
        x2_table,
    )


def _generate_u64_scale(shape):
    fp32_scale = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
    u32 = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    u64 = u32.astype(np.uint64)
    return u64
