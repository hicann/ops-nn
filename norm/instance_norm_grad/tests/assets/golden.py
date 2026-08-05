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

__golden__ = {"kernel": {"instance_norm_grad": "instance_norm_grad_golden"}}

INSTANCE_NORM_GRAD_EPS = 1e-6


def instance_norm_grad_golden(dy, x, variance, mean, gamma, **kwargs):
    """
    Golden for InstanceNormGrad. Inputs follow A2 order (dy, x, variance, mean, gamma).
    Layout is NDHWC: reduce over spatial dims (D,H,W) per (N, C) instance; gamma/beta grads
    additionally reduce over N (kept over C only). variance is RAW variance; rstd computed with
    fixed eps = 1e-6. All math in float64; do NOT rederive variance from a fresh forward.

    Returns: (pd_x, pd_gamma, pd_beta)
    """
    ori_dtype = dy.dtype
    nd = x.ndim
    C = x.shape[-1]
    reduce_axes = tuple(range(1, nd - 1))  # spatial axes (D,H,W)
    m = 1
    for ax in reduce_axes:
        m *= x.shape[ax]

    dyf = dy.astype(np.float64)
    xf = x.astype(np.float64)

    pshape = [x.shape[0]] + [1] * (nd - 2) + [C]  # [N,1,...,1,C]
    varb = variance.astype(np.float64).reshape(pshape)
    meanb = mean.astype(np.float64).reshape(pshape)
    gshape = [1] * nd
    gshape[-1] = C
    gammab = gamma.astype(np.float64).reshape(gshape)

    rstd = np.power(varb + INSTANCE_NORM_GRAD_EPS, -0.5)
    rstd3 = np.power(varb + INSTANCE_NORM_GRAD_EPS, -1.5)

    xc = xf - meanb
    pd_xl = dyf * gammab
    pd_var = np.sum(-0.5 * pd_xl * xc * rstd3, axis=reduce_axes, keepdims=True)
    pd_mean = np.sum(-1.0 * pd_xl * rstd, axis=reduce_axes, keepdims=True)
    # m == 0 means a *spatial* axis (D/H/W) is empty: there is nothing to average over, so the
    # 1/m correction terms do not exist. The kernel's empty branch (tilingKey 500) produces an
    # empty pd_x and zeroed pd_gamma/pd_beta; matching that here keeps spatial-zero cases
    # verifiable instead of crashing the golden with a division by zero.
    inv_m = 0.0 if m == 0 else 1.0 / m
    pd_x = pd_xl * rstd + pd_var * (2.0 * inv_m) * xc + pd_mean * inv_m

    x_hat = xc * rstd
    pd_gamma = np.sum(dyf * x_hat, axis=(0,) + reduce_axes)  # keep C
    pd_beta = np.sum(dyf, axis=(0,) + reduce_axes)  # keep C

    return (
        pd_x.astype(ori_dtype, copy=False),
        pd_gamma.astype(ori_dtype, copy=False),
        pd_beta.astype(ori_dtype, copy=False),
    )
