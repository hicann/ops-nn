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
"""
TTK golden plugin for batch_norm_ext2 (kernel mode, arch35/Ascend950).

**Implemented with torch tensor ops** (torch.mean / torch.rsqrt / addcmul), not hand-written
numpy formulas — the semantics were validated against tf.raw_ops.FusedBatchNormV2 on GPU:
  training : output_mean = batch mean (biased), output_variance = unbiased var (Bessel),
             reserve_space_1 = batch mean, reserve_space_2 = rstd = 1/sqrt(var+eps)
  inference: y = scale*(x-mean)*rsqrt(var+eps)+offset; the 4 stats outputs equal the
             input mean/variance (CANN compatibility behavior).

Reference for IO/attr semantics:
    ops-nn/norm/batch_norm_ext2/op_host/batch_norm_ext2_def.cpp
    ops-nn/norm/batch_norm_ext2/op_graph/batch_norm_ext2_proto.h

Canonical IO order:
    inputs : input_x(fp16|fp32, NCHW/NHWC 4D), input_scale(fp32, C), input_offset(fp32, C),
             input_mean(fp32, C, optional), input_variance(fp32, C, optional)
    outputs: output_y(like input_x), output_mean(fp32,C), output_variance(fp32,C),
             output_reserve_space_1(fp32,C), output_reserve_space_2(fp32,C)
    attrs  : epsilon(float=1e-4), data_format(str="NHWC"), is_training(bool=True)

TTK passes input arrays positionally in CSV input_shapes order; attributes via **kwargs.
In training mode input_mean/input_variance are ignored (may be None or dummy data).
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _f32(a):
    return np.asarray(a).astype(np.float32)


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def __golden_batch_norm_ext2(
    input_x, input_scale, input_offset, input_mean, input_variance, **kwargs
):
    eps = float(_attr(kwargs, "eps", _attr(kwargs, "epsilon", 1e-4)))
    data_format = str(_attr(kwargs, "data_format", "NHWC"))
    is_training = _as_bool(_attr(kwargs, "is_training", True), True)

    x = np.asarray(input_x)
    scale = _f32(input_scale).reshape(-1)
    offset = _f32(input_offset).reshape(-1)
    x_dtype = str(x.dtype)

    if data_format.upper() == "NHWC":
        axis = (0, 1, 2)
        num = int(np.prod(x.shape[:3]))
    else:  # NCHW
        axis = (0, 2, 3)
        num = int(np.prod([x.shape[0], x.shape[2], x.shape[3]]))

    xt = torch.from_numpy(_f32(x))
    st = torch.from_numpy(scale)
    ot = torch.from_numpy(offset)

    if is_training:
        # batch mean / biased var over spatial dims, per channel
        mean_t = torch.mean(xt, dim=axis, keepdim=True)
        var_t = torch.mean((xt - mean_t) ** 2, dim=axis, keepdim=True)
        rstd_t = torch.rsqrt(var_t + eps)
        # y = scale * (x - mean) * rstd + offset  (competing-op composition, same algorithm)
        y_t = torch.addcmul(
            ot.reshape(1, 1, 1, -1)
            if data_format.upper() == "NHWC"
            else ot.reshape(1, -1, 1, 1),
            (xt - mean_t) * rstd_t,
            st.reshape(1, 1, 1, -1)
            if data_format.upper() == "NHWC"
            else st.reshape(1, -1, 1, 1),
        )
        batch_mean = mean_t.reshape(-1).numpy()
        batch_var = var_t.reshape(-1).numpy()
        rstd = rstd_t.reshape(-1).numpy()
        # output_variance is unbiased (Bessel correction) — same as TF FusedBatchNormV2
        unbiased_var = (
            batch_var * (num / (num - 1)) if num > 1 else batch_var * float("inf")
        )
        out_mean = batch_mean
        out_variance = unbiased_var
        out_rs1 = batch_mean
        out_rs2 = rstd
    else:
        mean_in = _f32(input_mean).reshape(-1)
        var_in = _f32(input_variance).reshape(-1)
        mt = torch.from_numpy(mean_in)
        vt = torch.from_numpy(var_in)
        rstd_t = torch.rsqrt(vt + eps)
        if data_format.upper() == "NHWC":
            y_t = torch.addcmul(
                ot.reshape(1, 1, 1, -1),
                (xt - mt.reshape(1, 1, 1, -1)) * rstd_t.reshape(1, 1, 1, -1),
                st.reshape(1, 1, 1, -1),
            )
        else:  # NCHW
            y_t = torch.addcmul(
                ot.reshape(1, -1, 1, 1),
                (xt - mt.reshape(1, -1, 1, 1)) * rstd_t.reshape(1, -1, 1, 1),
                st.reshape(1, -1, 1, 1),
            )
        out_mean = mean_in
        out_variance = var_in
        out_rs1 = mean_in
        out_rs2 = var_in

    y = y_t.numpy().astype(x_dtype)
    return [
        y,
        out_mean.astype(np.float32),
        out_variance.astype(np.float32),
        out_rs1.astype(np.float32),
        out_rs2.astype(np.float32),
    ]


__golden__ = {"kernel": {"batch_norm_ext2": "__golden_batch_norm_ext2"}}
