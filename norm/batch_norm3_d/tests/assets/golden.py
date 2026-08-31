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

import numpy as np
import torch
import torch.nn.functional as F


__spec__ = {"batch_norm3d": "BatchNorm3DSpec"}
__golden__ = {"kernel": {"batch_norm3d": "batch_norm3d_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _to_ncdhw_torch(x, data_format):
    if data_format == "NDHWC":
        return x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def _from_ncdhw_torch(x, data_format):
    if data_format == "NDHWC":
        return x.permute(0, 2, 3, 4, 1).contiguous()
    return x


class _BatchNorm3DCompose:
    """Third-party benchmark on GPU, using PyTorch batch_norm and casting outputs
    back to the NPU output dtypes for a fair cross_check comparison."""

    def __init__(self, epsilon=0.0001, data_format="NCDHW", is_training=True, **_):
        self.epsilon = epsilon
        self.data_format = data_format
        self.is_training = is_training

    def __call__(self, x, scale, offset, mean=None, variance=None, **_):
        x_n = _to_ncdhw_torch(x, self.data_format)
        scale_f = scale.to(torch.float32)
        offset_f = offset.to(torch.float32)
        if self.is_training:
            y_n = F.batch_norm(
                x_n.to(torch.float32),
                None,
                None,
                scale_f,
                offset_f,
                training=True,
                momentum=0.1,
                eps=self.epsilon,
            )
            axes = (0, 2, 3, 4)
            batch_mean = x_n.to(torch.float32).mean(dim=axes)
            batch_var = x_n.to(torch.float32).var(dim=axes, unbiased=False)
            reduce_count = 1
            for axis in axes:
                reduce_count *= x_n.shape[axis]
            saved_var = batch_var * (float(reduce_count) / float(reduce_count - 1))
            saved_rstd = torch.rsqrt(batch_var + self.epsilon)
        else:
            batch_mean = mean.to(torch.float32)
            batch_var = variance.to(torch.float32)
            view_shape = [1, x_n.shape[1], 1, 1, 1]
            y_n = (
                x_n.to(torch.float32) - batch_mean.reshape(view_shape)
            ) * torch.rsqrt(batch_var.reshape(view_shape) + self.epsilon)
            y_n = y_n * scale_f.reshape(view_shape) + offset_f.reshape(view_shape)
            saved_var = batch_var
            saved_rstd = batch_var
        y = _from_ncdhw_torch(y_n, self.data_format).to(x.dtype)
        return [y, batch_mean, saved_var, batch_mean.clone(), saved_rstd]


class BatchNorm3DSpec:
    @staticmethod
    def golden(x, scale, offset, mean=None, variance=None, **kwargs):
        return batch_norm3d_golden(x, scale, offset, mean, variance, **kwargs)

    third_party = {"torch": _BatchNorm3DCompose}
    tolerance = _TOL


def batch_norm3d_golden(
    x,
    scale,
    offset,
    mean=None,
    variance=None,
    *,
    epsilon=0.0001,
    data_format="NCDHW",
    is_training=True,
    **kwargs,
):
    """Golden for BatchNorm3D (torch implementation, float64 intermediate).

    Training mode follows PyTorch/GPU batch_norm behavior. The returned save_invstd
    is rstd = 1 / sqrt(batch_var + epsilon), which is the value consumed by
    BatchNorm3DGrad/BatchNormGradExt2 reserve_space_2 in training mode.
    """
    x_in = torch.from_numpy(np.ascontiguousarray(x))
    x_n = _to_ncdhw_torch(x_in, data_format).to(torch.float64)
    scale_f = torch.from_numpy(np.ascontiguousarray(scale)).to(torch.float64)
    offset_f = torch.from_numpy(np.ascontiguousarray(offset)).to(torch.float64)

    if is_training:
        axes = (0, 2, 3, 4)
        batch_mean = x_n.mean(dim=axes)
        batch_var = x_n.var(dim=axes, unbiased=False)
        reduce_count = 1
        for axis in axes:
            reduce_count *= x_n.shape[axis]
        saved_var = batch_var * (float(reduce_count) / float(reduce_count - 1))
        saved_rstd = torch.rsqrt(batch_var + epsilon)
        y_t = F.batch_norm(
            x_n, None, None, scale_f, offset_f, training=True, momentum=0.1, eps=epsilon
        )
    else:
        batch_mean = torch.from_numpy(np.ascontiguousarray(mean)).to(torch.float64)
        batch_var = torch.from_numpy(np.ascontiguousarray(variance)).to(torch.float64)
        saved_var = batch_var
        saved_rstd = batch_var
        view_shape = [1, x_n.shape[1], 1, 1, 1]
        y_t = (x_n - batch_mean.view(view_shape)) * torch.rsqrt(
            batch_var.view(view_shape) + epsilon
        )
        y_t = y_t * scale_f.view(view_shape) + offset_f.view(view_shape)

    y = _from_ncdhw_torch(y_t, data_format).to(x_in.dtype).numpy()
    return [
        y,
        batch_mean.to(torch.float32).numpy(),
        saved_var.to(torch.float32).numpy(),
        batch_mean.to(torch.float32).numpy(),
        saved_rstd.to(torch.float32).numpy(),
    ]
