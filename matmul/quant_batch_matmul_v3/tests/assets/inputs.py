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
        "kernel": {
            "quant_batch_matmul_v3": "quant_batch_matmul_v3_inputs"
        }
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
            orig_range = input_ranges[input_index] if input_index < len(input_ranges) else None
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


def quant_batch_matmul_v3_inputs(x1, x2, scale, offset = None, bias = None, pertoken_scale = None, *, dtype: int,
                                 transpose_x1: bool = False, transpose_x2: bool = False,
                                 group_size:int = 0, **kwargs):
    input_deq_scale = scale
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    testcase_name = kwargs.get('testcase_name', 'unknown')
    input_ranges = kwargs.get('input_ranges', None)

    x1 = _check_and_fix_unsigned_dtype_nan(x1, 0, input_ranges, testcase_name)
    x2 = _check_and_fix_unsigned_dtype_nan(x2, 1, input_ranges, testcase_name)
    input_deq_scale = _check_and_fix_unsigned_dtype_nan(input_deq_scale, 2, input_ranges, testcase_name)
    pertoken_scale = _check_and_fix_unsigned_dtype_nan(pertoken_scale, 5, input_ranges, testcase_name)

    if input_deq_scale.dtype in ("uint64", "int64") and pertoken_scale is None:
        deq_scale_shape = scale.shape
        target_dtype = input_deq_scale.dtype
        input_deq_scale = scale_generate(deq_scale_shape, offset, out_dtype)
        if target_dtype == "int64":
            input_deq_scale = input_deq_scale.astype(np.int64)

    return x1, x2, input_deq_scale, offset, bias, pertoken_scale

def scale_generate(deq_scale_shape, offset, out_dtype):
    has_offset = offset is not None

    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
    uint32_deq_scale &= 0XFFFFE000

    if has_offset:
        offset_shape = offset.shape
        fp32_offset = np.random.uniform(low=-5, high=5, size=offset_shape).astype(np.float32)

    if out_dtype != "int8":
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
    elif out_dtype == "int8":
        s9_offset = 0
        if has_offset:
            s9_offset = f32_2_s9(fp32_offset).astype(int).reshape(offset_shape)
            s9_offset &= 0X1FF
            s9_offset = s9_offset[0] if deq_scale_shape[-1] < offset_shape[-1] else s9_offset
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        uint64_deq_scale |= np.uint64(s9_offset << 37)
        uint64_deq_scale |= 1 << 46
    return uint64_deq_scale

def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip