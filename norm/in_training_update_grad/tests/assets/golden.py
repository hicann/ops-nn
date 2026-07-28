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


__golden__ = {"kernel": {"in_training_update_grad": "in_training_update_grad_golden"}}


def in_training_update_grad_golden(
    dy,
    x,
    variance,
    mean,  # inputs (NDC1HWC0, 6D)
    **kwargs,
):
    """
    Golden for INTrainingUpdateGrad (InstanceNorm training backward, reduce-over-spatial stage).
    Parameter names/order follow in_training_update_grad_def.cpp (inputs only).
    All input Tensors are numpy.ndarray in NDC1HWC0 (N, D, C1, H, W, C0).

    x_norm    = (x - mean) * rsqrt(variance + 1e-6)     # mean/variance broadcast over D,H,W
    res_gamma = sum over (D, H, W) of dy * x_norm        # keepdims, spatial dims -> 1
    res_beta  = sum over (D, H, W) of dy                 # keepdims
    An empty reduction (D/H/W == 0) yields 0.0 for both outputs (sum over an empty set).
    High-precision (fp64) reference: the kernel accumulates in fp32 with Kahan compensation and is
    accurate to the fp32 floor, so the golden must be the fp64 truth (competitor ops stitched in
    fp64) — an fp32 golden carries its own ~eps*kappa reduction error on large/cancelling reductions
    (e.g. D=80000) and would falsely flag the (more accurate) kernel. Matches instance_norm_grad's
    fp64 golden convention.
    """
    import torch

    eps = 1e-6
    reduce_axes = (1, 3, 4)  # D, H, W

    dy_t = torch.from_numpy(np.ascontiguousarray(dy).astype(np.float64))
    x_t = torch.from_numpy(np.ascontiguousarray(x).astype(np.float64))
    var_t = torch.from_numpy(np.ascontiguousarray(variance).astype(np.float64))
    mean_t = torch.from_numpy(np.ascontiguousarray(mean).astype(np.float64))

    rstd = torch.rsqrt(var_t + eps)
    x_norm = (x_t - mean_t) * rstd
    res_gamma = (dy_t * x_norm).sum(dim=reduce_axes, keepdim=True)
    res_beta = dy_t.sum(dim=reduce_axes, keepdim=True)

    return res_gamma.numpy().astype(np.float64), res_beta.numpy().astype(np.float64)
