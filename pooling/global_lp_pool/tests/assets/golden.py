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

__golden__ = {"kernel": {"global_lp_pool": "global_lp_pool_golden"}}


def global_lp_pool_golden(x, **kwargs):
    """
    y = (sum(|x|^p))^(1/p) along spatial dims.
    """
    p = kwargs.get("p", 2.0)
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    input_dtype = x.dtype

    # Convert to float32 for computation
    x_f32 = x.astype(np.float32)

    # Spatial axes: dims[2:]
    ndim = x.ndim
    spatial_axes = tuple(range(2, ndim))

    # |x|^p
    powered = np.power(np.abs(x_f32), p)

    # Sum along spatial dims
    summed = np.sum(powered, axis=spatial_axes, keepdims=True)

    # (sum)^(1/p)
    result = np.power(summed, 1.0 / p)

    # Cast back to original dtype
    if "bfloat16" in str(ori_dtype).lower():
        return result.astype(input_dtype, copy=False)
    return result.astype(input_dtype, copy=False)
