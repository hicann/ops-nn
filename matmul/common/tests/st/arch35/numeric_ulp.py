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
"""
numeric ULP comparison (standalone, no ttk dependency).
"""

import ast
import math

import numpy as np

import matmul_golden_util as _util

torch_to_numpy = _util.torch_to_numpy


def _to_float(arr):
    dtype_str = str(arr.dtype).split(".")[-1]
    if "float4" in dtype_str:
        return arr.astype(np.float32, copy=False)
    if "float8_e8m0" in dtype_str:
        return arr.view(np.uint8).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def numeric_ulp_compare(npu_out, golden_out, compare_context=None, **kwargs):
    """基于 ULP 的自定义精度比较。

    将输出转为 float32 后逐元素比较绝对误差，阈值固定 1.0。

    Args:
        npu_out: NPU 输出数组
        golden_out: golden 参考数组
        compare_context: ttk 注入的 CompareContext（保留用于签名兼容，当前未使用）
        **kwargs: 吸收 ttk 传入的额外参数

    Returns:
        dict: {"pass": bool, "precision": float}
    """
    if hasattr(npu_out, "detach"):
        npu_out = torch_to_numpy(npu_out)
    if hasattr(golden_out, "detach"):
        golden_out = torch_to_numpy(golden_out)

    output = _to_float(npu_out)
    golden = _to_float(golden_out)

    output_flat = output.ravel()
    golden_flat = golden.ravel()
    diff_flat = np.abs(np.subtract(output_flat, golden_flat))
    diff_mask = diff_flat > 1.0
    diff_indices = np.where(diff_mask)[0]

    npu_nan = np.isnan(output_flat)
    golden_nan = np.isnan(golden_flat)
    both_nan = np.logical_and(npu_nan, golden_nan)
    diff_indices = np.setdiff1d(diff_indices, np.where(both_nan)[0])

    golden_size = golden_flat.size
    diff_size = diff_indices.size
    if golden_size == 0:
        return {"pass": diff_size == 0, "precision": 100.0}

    csv = compare_context.csv_fields if compare_context else {}
    ptol_raw = csv.get("precision_tolerances", "")
    if ptol_raw:
        _, ptol = ast.literal_eval(ptol_raw)[0]
    else:
        dtype_str = str(npu_out.dtype).split(".")[-1]
        ptol = 0.001 if dtype_str != "float32" else 0.0001

    precision = math.floor((golden_size - diff_size) / golden_size * 10000) / 100
    is_pass = diff_size / golden_size <= ptol
    return {"pass": bool(is_pass), "precision": precision}
