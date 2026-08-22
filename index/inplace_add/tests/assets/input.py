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

from math import gcd

import numpy as np

__input__ = {"kernel": {"inplace_add": "inplace_add_input"}}


def inplace_add_input(x, indices, v, **kwargs):
    """Generate unique numerical indices or explicit safety-only conflicts."""
    testcase_name = kwargs.get("testcase_name", "")
    if indices.size == 0 or x.shape[0] == 0:
        return [x, indices, v]

    n = x.shape[0]
    conflict_safety_only = "conflict_safety" in testcase_name
    if "explicit_duplicate" in testcase_name:
        values = np.zeros(indices.size, dtype=indices.dtype)
    elif "mod_collision" in testcase_name:
        values = np.arange(indices.size, dtype=np.int64)
        values[1::2] += n
        values = values.astype(indices.dtype, copy=False)
    elif "k_gt_n" in testcase_name:
        values = np.arange(indices.size, dtype=indices.dtype)
    elif "integer_endpoints" in testcase_name:
        values = np.array([0, 1], dtype=indices.dtype)
    elif "int32_limits" in testcase_name:
        values = np.array(
            [np.iinfo(np.int32).min, np.iinfo(np.int32).max],
            dtype=indices.dtype,
        )
    elif "modulo" in testcase_name:
        values = np.array([-1, n + 1], dtype=indices.dtype)
    elif "negative" in testcase_name:
        values = np.array([-1, -n], dtype=indices.dtype)
    elif "wrapped" in testcase_name:
        values = np.array([n, 2 * n + 1], dtype=indices.dtype)
    elif "reverse" in testcase_name:
        values = np.arange(indices.size - 1, -1, -1, dtype=indices.dtype)
    elif "first_last" in testcase_name:
        values = np.array([0, n - 1], dtype=indices.dtype)
    elif "spread" in testcase_name:
        step = 2
        while step < n and gcd(step, n) != 1:
            step += 1
        if step >= n:
            step = 1
        values = (np.arange(indices.size, dtype=np.int64) * step) % n
        values = values.astype(indices.dtype, copy=False)
    else:
        if indices.size > n:
            raise ValueError(
                "InplaceAdd numerical cases require K <= N so normalized rows stay unique."
            )
        values = np.arange(indices.size, dtype=indices.dtype)

    indices = np.resize(values, indices.size).astype(indices.dtype, copy=False)
    indices_i64 = indices.astype(np.int64, copy=False)
    normalized = ((indices_i64 % n) + n) % n
    if not conflict_safety_only and np.unique(normalized).size != indices.size:
        raise ValueError(
            f"{testcase_name}: normalized indices must be unique for numerical validation."
        )

    if "integer_endpoints" in testcase_name:
        dtype_info = np.iinfo(x.dtype)
        x = np.zeros_like(x)
        v = np.zeros_like(v)
        x.reshape(x.shape[0], -1)[0] = dtype_info.min
        v.reshape(v.shape[0], -1)[1] = dtype_info.max
    elif "x_zero" in testcase_name:
        x = np.zeros_like(x)
    if "v_zero" in testcase_name:
        v = np.zeros_like(v)
    elif "nan" in testcase_name:
        v = np.zeros_like(v)
        v.reshape(-1)[0] = complex(np.nan, np.nan) if np.iscomplexobj(v) else np.nan
    elif "pos_inf" in testcase_name:
        v = np.zeros_like(v)
        v.reshape(-1)[0] = complex(np.inf, np.inf) if np.iscomplexobj(v) else np.inf
    elif "neg_inf" in testcase_name:
        v = np.zeros_like(v)
        v.reshape(-1)[0] = complex(-np.inf, -np.inf) if np.iscomplexobj(v) else -np.inf
    elif "fp16_overflow" in testcase_name:
        x = np.full_like(x, 60000.0)
        v = np.full_like(v, 60000.0)
    return [x, indices.reshape(indices.shape), v]
