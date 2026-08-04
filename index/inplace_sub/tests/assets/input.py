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

import hashlib

import numpy as np

__input__ = {"kernel": {"inplace_sub": "inplace_sub_input"}}


def _rng_for_case(testcase_name):
    digest = hashlib.md5(testcase_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _supports_special_values(array):
    dtype_name = str(array.dtype)
    return (
        "bfloat16" in dtype_name
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    )


def inplace_sub_input(x, indices, v, **kwargs):
    testcase_name = kwargs.get("testcase_name", "")
    if "dfx_nan" in testcase_name and _supports_special_values(x):
        x = np.array(x, copy=True)
        v = np.array(v, copy=True)
        if x.size > 0:
            x.reshape(-1)[0] = np.nan
        if v.size > 0:
            v.reshape(-1)[0] = np.nan
    elif "dfx_inf" in testcase_name and _supports_special_values(x):
        x = np.array(x, copy=True)
        v = np.array(v, copy=True)
        if x.size > 0:
            x.reshape(-1)[0] = np.inf
        if v.size > 0:
            v.reshape(-1)[0] = -np.inf

    if indices.size == 0 or x.shape[0] == 0:
        return [x, indices, v]
    indices_shape = indices.shape
    if "dfx_negative_index" in testcase_name:
        values = np.array([-1, -2, -x.shape[0]], dtype=indices.dtype)
        indices = np.resize(values, indices.size).astype(indices.dtype)
        return [x, indices.reshape(indices_shape), v]
    if "dfx_wrapped_index" in testcase_name:
        values = np.array(
            [x.shape[0], x.shape[0] + 1, 2 * x.shape[0] - 1], dtype=indices.dtype
        )
        indices = np.resize(values, indices.size).astype(indices.dtype)
        return [x, indices.reshape(indices_shape), v]
    if indices.size > x.shape[0]:
        raise ValueError(
            "InplaceSub positive cases require indices length not greater than x.shape[0]."
        )
    rng = _rng_for_case(testcase_name)
    indices = rng.choice(x.shape[0], indices.size, replace=False).astype(indices.dtype)
    return [x, indices.reshape(indices_shape), v]
