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

__golden__ = {"kernel": {"l2_normalize_grad": "l2_normalize_grad_golden"}}


def _resolve_axis(dim, rank):
    if dim is None:
        return 1
    if isinstance(dim, (list, tuple, np.ndarray)):
        vals = list(dim)
        axis = int(vals[0]) if len(vals) > 0 else 1
    else:
        axis = int(dim)
    if axis < 0:
        axis += rank
    return axis


def l2_normalize_grad_golden(x, y, dy, *, dim=(1,), eps=1e-4, **kwargs):
    """
    Golden for L2NormalizeGrad. Args (names/order) follow l2_normalize_grad_def.cpp inputs.
    All input tensors are numpy.ndarray. Returns dx (same shape/dtype as x).

    Formula source is the **ascend910b algorithm spec** (l2_normalize_grad.py), NOT the kernel under
    test. What is aligned with the kernel is only the *input contract*: both consume the provided y
    directly instead of recomputing it from x, so arbitrary (x, y, dy) triples that are not
    self-consistent are handled identically -- otherwise every random case would be a false failure.
    Same convention as rms_norm_grad's golden consuming the provided rstd:
        n  = max(sqrt(sum(x*x, dim)), eps)
        s  = sum(y*dy, dim)
        dx = (dy - y*s) / n
    When y == F.normalize(x, p=2, dim, eps) (consistent, normal-scale inputs so ||x|| > eps), this
    equals the torch autograd of F.normalize through y.backward(dy) (see 00_spec 2 / 6.1).
    High-precision (fp64) reference: golden = competitor formula stitched in fp64 (the truth); the
    fp32 kernel is compared to it under the CANN atol+rtol/percent standard. An fp32 golden would
    carry its own fp32 reduction/cancellation error (dx = dy - y*s is a cancelling difference) and
    could falsely flag the kernel. Matches instance/in_training fp64 golden convention.
    """
    axis = _resolve_axis(dim, x.ndim)
    eps_f = np.float64(eps)

    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    dyf = dy.astype(np.float64)

    sq = np.sum(xf * xf, axis=axis, keepdims=True)
    n = np.maximum(np.sqrt(sq), eps_f)
    s = np.sum(yf * dyf, axis=axis, keepdims=True)
    dx = (dyf - yf * s) / n

    return dx.astype(np.float64, copy=False)
