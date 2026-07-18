#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK custom golden for deep_norm (DeepNorm).

Compute formula (docs/aclnnDeepNorm.md / op_host/deep_norm_def.cpp):
    h    = alpha * x + gx
    mean = mean(h, axis=-1, keepdims)            # over last len(gamma) dims (gamma is 1D -> last dim)
    var  = mean((h - mean)^2, axis=-1, keepdims)
    rstd = 1 / sqrt(var + epsilon)
    y    = (h - mean) * rstd * gamma + beta

Inputs (order, matches op_def):   x, gx, beta, gamma
    x, gx : shape [..., D]   (same shape/dtype)
    beta, gamma : shape [D]
Outputs (order, matches op_def):  mean, rstd, y
    mean, rstd : shape [..., 1] (broadcast: leading dims kept, norm dims -> 1), ALWAYS float32
    y          : shape [..., D], same dtype as x

All arithmetic is done in float32 (b16 upcast). mean/rstd stay float32;
y is cast back to x's dtype (declared per-output in output_dtypes).
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

ALPHA_DEFAULT = 0.3
EPS_DEFAULT = 1e-6


def _get_attrs(kwargs):
    attrs = kwargs.get("attributes") or {}
    if isinstance(attrs, str):
        # attributes may arrive as a string (e.g. '{"alpha": 0.3, "epsilon": 1e-06}')
        import ast

        try:
            attrs = ast.literal_eval(attrs)
        except Exception:
            attrs = {}
    if not isinstance(attrs, dict):
        attrs = {}
    alpha = float(attrs.get("alpha", ALPHA_DEFAULT))
    eps = float(attrs.get("epsilon", EPS_DEFAULT))
    return alpha, eps


def _cast(arr32, target):
    target = str(target)
    if target == "bfloat16":
        return arr32.astype(_bf16) if _bf16 is not None else arr32
    return arr32.astype(target)


def __golden_deep_norm(x, gx, beta, gamma, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    alpha, eps = _get_attrs(kwargs)

    x32 = np.asarray(x).astype(np.float32)
    gx32 = np.asarray(gx).astype(np.float32)
    beta32 = np.asarray(beta).astype(np.float32)
    gamma32 = np.asarray(gamma).astype(np.float32)

    # reduce over the last len(gamma) dims (gamma is 1D -> last dim).
    gn = gamma32.ndim
    reduce_axis = tuple(range(x32.ndim - gn, x32.ndim))

    h = alpha * x32 + gx32
    mean = np.mean(h, axis=reduce_axis, keepdims=True)
    diff = h - mean
    var = np.mean(diff * diff, axis=reduce_axis, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = diff * rstd * gamma32 + beta32

    # per-output dtype: [mean, rstd, y]
    y_dt = "float32"
    if output_dtypes is not None and len(output_dtypes) >= 3:
        y_dt = str(output_dtypes[2])
    else:
        y_dt = str(np.asarray(x).dtype)

    mean_out = mean.astype(np.float32)
    rstd_out = rstd.astype(np.float32)
    y_out = _cast(y, y_dt)
    return [mean_out, rstd_out, y_out]


__golden__ = {"kernel": {"deep_norm": "__golden_deep_norm"}}
