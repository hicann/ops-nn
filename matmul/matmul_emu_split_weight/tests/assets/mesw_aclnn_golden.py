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
"""ACLNN 层 golden 实现。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import mesw_kernel_golden as _kernel


def _to_np(*tensors):
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        else:
            result.append(_util.torch_to_numpy(t))
    return tuple(result)


def _out_dtype(out, fallback):
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _detect_transpose(tensor, kwargs, key):
    explicit = kwargs.get(key)
    if explicit is not None:
        return bool(explicit)
    return _util.detect_transpose_from_strides(tensor)


class AclnnMatmulEmuSplitWeightTestSpec:
    @staticmethod
    def golden(x, w_high, w_low, out=None, w_low_scale=0.00390625, y_dtype=0, **kwargs):
        x_np, w_high_np, w_low_np = _to_np(x, w_high, w_low)
        transpose_x = _detect_transpose(x, kwargs, "transpose_x")
        transpose_w = _detect_transpose(w_high, kwargs, "transpose_w")
        out_dtype = _out_dtype(out, x)
        temp_kwargs = dict(kwargs)
        if out_dtype:
            temp_kwargs["output_dtypes"] = [out_dtype]
        return _kernel.MatmulEmuSplitWeightKernelTestSpec.golden(
            x_np,
            w_high_np,
            w_low_np,
            transpose_x=transpose_x,
            transpose_w=transpose_w,
            scale=w_low_scale,
            y_dtype=y_dtype,
            **temp_kwargs,
        )[0]


__spec__ = {
    "aclnnMatmulEmuSplitWeight": "AclnnMatmulEmuSplitWeightTestSpec",
}
