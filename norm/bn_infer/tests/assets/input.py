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


__input__ = {"kernel": {"bn_infer": "bn_infer_input"}}


def _supports_special_values(array):
    dtype_name = str(array.dtype)
    return (
        "bfloat16" in dtype_name
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    )


def bn_infer_input(x, scale, offset, mean, variance, **kwargs):
    testcase_name = kwargs.get("testcase_name", "")
    x = np.array(x, copy=True)
    scale = np.array(scale, copy=True)
    offset = np.array(offset, copy=True)
    mean = np.array(mean, copy=True)
    variance = np.array(variance, copy=True)

    if "dfx_zero_var" in testcase_name and variance.size > 0:
        variance.reshape(-1)[:] = 0
    if "dfx_nan" in testcase_name and _supports_special_values(x):
        if x.size > 0:
            x.reshape(-1)[0] = np.nan
        if mean.size > 0:
            mean.reshape(-1)[0] = np.nan
    if "dfx_inf" in testcase_name and _supports_special_values(x):
        if x.size > 0:
            x.reshape(-1)[0] = np.inf

    return [x, scale, offset, mean, variance]
