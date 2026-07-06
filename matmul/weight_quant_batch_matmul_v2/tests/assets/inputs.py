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
__input__ = {
    "kernel": {"weight_quant_batch_matmul_v2": "weight_quant_batch_matmul_v2_inputs"}
}

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


def weight_quant_batch_matmul_v2_inputs(
    x1,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    *,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    dtype: int = -1,
    inner_precise: int = 0,
    **kwargs,
):
    output_dtypes = kwargs.get("output_dtypes", ["float16"])
    input_ranges = kwargs.get("input_ranges", None)
    testcase_name = kwargs.get("testcase_name", "unknown")

    antiquant_scale = _check_and_fix_unsigned_dtype_nan(
        antiquant_scale, 2, input_ranges, testcase_name
    )

    if antiquant_offset is not None:
        offset_range = _get_input_range(input_ranges, 3)
        antiquant_offset = change_antiquant_dtype(
            antiquant_offset, x1.dtype, offset_range
        )

    if antiquant_scale.dtype.name in ("uint64", "int64"):
        target_dtype = antiquant_scale.dtype
        antiquant_scale = _antiquant_scale_generate(
            antiquant_scale, input_ranges, testcase_name
        )
        if target_dtype.name == "int64":
            antiquant_scale = antiquant_scale.astype(np.int64)

    if output_dtypes[0] == "int8":
        quant_scale = _quant_scale_generate(quant_scale.shape)

    return (
        x1,
        weight,
        antiquant_scale,
        antiquant_offset,
        quant_scale,
        quant_offset,
        bias,
    )


def _get_input_range(input_ranges, index):
    if input_ranges is None:
        return None
    try:
        if index < len(input_ranges):
            rng = input_ranges[index]
            if rng is not None and len(rng) >= 2:
                return rng[0], rng[1]
    except (TypeError, IndexError):
        pass
    return None


# 伪量化算子量化参数dtype虽然是浮点, data_range要求是整型[-128,127]
def change_antiquant_dtype(antiquant_offset, x_dtype, input_data_range=None):
    input_range_left, input_range_right = -128, 127
    if input_data_range is not None:
        input_range_left = int(np.ceil(input_data_range[0]))
        input_range_right = int(np.ceil(input_data_range[1]))
    if input_range_right > 127:
        input_range_right = 127
    if input_range_left < -128:
        input_range_left = -128
    if input_range_left == input_range_right:
        if x_dtype.name in ("float16", "bfloat16"):
            antiquant_offset = np.full(
                antiquant_offset.shape, input_range_left, dtype=x_dtype
            )
    else:
        if x_dtype.name in ("float16", "bfloat16"):
            antiquant_offset = np.random.randint(
                input_range_left, input_range_right, size=antiquant_offset.shape
            ).astype(x_dtype)
    return antiquant_offset


def _antiquant_scale_generate(antiquant_scale, input_ranges, testcase_name):
    scale_shape = antiquant_scale.shape
    input_range_left, input_range_right = -65504, 65504
    if input_ranges is not None:
        try:
            if len(input_ranges) >= 3:
                rng = input_ranges[2]
            elif len(input_ranges) >= 2:
                rng = input_ranges[1]
            else:
                rng = input_ranges[0]
            input_range_left, input_range_right = rng[0], rng[1]
        except (TypeError, IndexError):
            pass
    input_range_left, input_range_right = process_quant_inf_nan(
        input_range_left, input_range_right, -65504, 65504
    )
    fp32_scale = (
        np.random.uniform(
            low=input_range_left, high=input_range_right, size=scale_shape
        )
        .astype(np.float16)
        .astype(np.float32)
    )
    return _pack_u64_scale(fp32_scale, None)


def _quant_scale_generate(scale_shape):
    fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    fp32_offset = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    return _pack_u64_scale(fp32_scale, fp32_offset)


def _pack_u64_scale(fp32_scale, fp32_offset=None):
    u32_scale = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
    u64 = u32_scale.astype(np.uint64)
    u64 |= np.uint64(1 << 46)
    if fp32_offset is not None:
        s9 = np.clip(np.round(fp32_offset), -256, 255).astype(np.int64)
        s9 = s9.astype(np.uint64) & np.uint64(0x1FF)
        u64 |= s9 << np.uint64(37)
    return u64


def process_quant_inf_nan(input_range_left, input_range_right, low, high):
    if "-inf" in str(input_range_left):
        input_range_left = low
    elif "inf" in str(input_range_left):
        input_range_left = high
    elif "nan" in str(input_range_left):
        input_range_left = 0

    if "-inf" in str(input_range_right):
        input_range_right = low
    elif "inf" in str(input_range_right):
        input_range_right = high
    elif "nan" in str(input_range_right):
        input_range_right = 0

    return input_range_left, input_range_right
