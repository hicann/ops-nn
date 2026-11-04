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
"""
rotate_quant ACLNN V2 golden 实现，覆盖普通与 M 轴分组场景。

aclnn 接口参数名使用 camelCase（与 C API 一致），golden 委托 kernel golden 的 compute 函数。
"""

import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

import rotate_quant_kernel_golden as _kernel  # noqa: E402
from rotate_quant_kernel_golden import compare_outputs  # noqa: E402

compute_rotate_mx = _kernel.compute_rotate_mx
compute_grouped_rotate_mx = _kernel.compute_grouped_rotate_mx

_DTYPE_TO_MX = {
    "float4_e2m1": "float4_e2m1",
    "float4": "float4_e2m1",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}


def mx_output_dtype(y_out, tensor_dtypes=()):
    candidates = [y_out.dtype] if y_out is not None and hasattr(y_out, "dtype") else []
    candidates.extend(tensor_dtypes or ())
    for dtype in candidates:
        name = str(dtype).split(".")[-1].lower()
        if name in _DTYPE_TO_MX:
            return _DTYPE_TO_MX[name]
        if name == "float4_e2m1fn":
            return "float4_e2m1"
    raise ValueError(
        "Cannot determine RotateQuant output dtype from yOut or tensor_dtypes"
    )


class AclnnRotateQuantV2TestSpec:
    """ACLNN V2 reference for ordinary and M-axis grouped RotateQuant."""

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        if compare_context is None:
            raise ValueError("ACLNN V2 comparison requires TTK compare_context")
        return compare_outputs(*outputs, compare_context=compare_context)

    @staticmethod
    def golden(
        x,
        rotation,
        alpha=None,
        groupList=None,
        groupListType=0,
        axis=-1,
        roundMode="rint",
        scaleAlg=0,
        dstTypeMax=0.0,
        trans=False,
        yOut=None,
        scaleOut=None,
        **kwargs,
    ):
        if groupListType not in (0, 1):
            raise ValueError("groupListType must be 0 or 1")
        if (axis is not None and axis not in (-1, x.ndim - 1)) or trans:
            raise ValueError("RotateQuant requires the last axis and trans=False")
        quant_options = {
            "mx_ele_dtype": mx_output_dtype(yOut, kwargs.get("tensor_dtypes", ())),
            "round_mode": "rint" if roundMode is None else str(roundMode),
            "scale_alg": 0 if scaleAlg is None else int(scaleAlg),
            "dst_type_max": 0.0 if dstTypeMax is None else float(dstTypeMax),
        }
        if groupList is None:
            return compute_rotate_mx(x, rotation, alpha, **quant_options)
        if yOut is None or scaleOut is None:
            raise ValueError("Grouped golden requires initial yOut and scaleOut")
        return compute_grouped_rotate_mx(
            x,
            rotation,
            alpha,
            groupList,
            groupListType,
            initial_y=yOut,
            initial_scale=scaleOut,
            **quant_options,
        )


__spec__ = {
    "aclnnRotateQuantV2": "AclnnRotateQuantV2TestSpec",
}
