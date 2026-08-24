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


__spec__ = {"batch_norm3d_grad": "BatchNorm3DGradSpec"}
__golden__ = {"kernel": {"batch_norm3d_grad": "batch_norm3d_grad_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_nchw_like_torch(x, data_format):
    if data_format == "NHWC":
        return x.permute(0, 3, 1, 2).contiguous()
    if data_format == "NDHWC":
        return x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def _from_nchw_like_torch(x, data_format):
    if data_format == "NHWC":
        return x.permute(0, 2, 3, 1).contiguous()
    if data_format == "NDHWC":
        return x.permute(0, 2, 3, 4, 1).contiguous()
    return x


def _channel_axes(rank):
    return tuple(i for i in range(rank) if i != 1)


class _BatchNorm3DGradCompose:
    """Third-party GPU implementation using the same BN backward formula as the
    operator contract. Outputs are cast to the NPU output dtypes."""

    def __init__(self, epsilon=0.0001, data_format="NCDHW", is_training=True, **_):
        self.epsilon = epsilon
        self.data_format = data_format
        self.is_training = is_training

    def __call__(self, y_backprop, x, scale, reserve_space_1, reserve_space_2, **_):
        x_n = _to_nchw_like_torch(x, self.data_format)
        dy_n = _to_nchw_like_torch(y_backprop, self.data_format)
        axes = _channel_axes(x_n.dim())
        reduce_count = 1
        for axis in axes:
            reduce_count *= x_n.shape[axis]
        broadcast_shape = [1] * x_n.dim()
        broadcast_shape[1] = x_n.shape[1]

        scale_n = scale.to(torch.float32).reshape(broadcast_shape)
        mean_n = reserve_space_1.to(torch.float32).reshape(broadcast_shape)
        if self.is_training:
            rstd = reserve_space_2.to(torch.float32)
        else:
            rstd = torch.rsqrt(reserve_space_2.to(torch.float32) + self.epsilon)
        rstd_n = rstd.reshape(broadcast_shape)

        dy_f = dy_n.to(torch.float32)
        x_hat = (x_n.to(torch.float32) - mean_n) * rstd_n
        dscale = torch.sum(dy_f * x_hat, dim=axes).to(torch.float32)
        doffset = torch.sum(dy_f, dim=axes).to(torch.float32)
        if self.is_training:
            dscale_n = dscale.reshape(broadcast_shape)
            doffset_n = doffset.reshape(broadcast_shape)
            dx_n = (
                (dy_f - doffset_n / reduce_count - x_hat * dscale_n / reduce_count)
                * scale_n
                * rstd_n
            )
        else:
            dx_n = dy_f * scale_n * rstd_n
        dx = _from_nchw_like_torch(dx_n, self.data_format).to(x.dtype)
        empty = torch.empty((0,), device=x.device, dtype=torch.float32)
        return [dx, dscale, doffset, empty, empty]


class BatchNorm3DGradSpec:
    @staticmethod
    def golden(y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs):
        return batch_norm3d_grad_golden(
            y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs
        )

    third_party = {"torch": _BatchNorm3DGradCompose}
    tolerance = _TOL


def batch_norm3d_grad_golden(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    *,
    epsilon=0.0001,
    data_format="NCDHW",
    is_training=True,
    **kwargs,
):
    """Golden for BatchNorm3DGrad (torch implementation, float64 intermediate).

    In training mode, reserve_space_2 is save_invstd/rstd, not variance. In
    inference mode, reserve_space_2 is variance. This matches the operator
    interface and PyTorch/GPU BN backward mathematics.
    """
    x_in = torch.from_numpy(np.ascontiguousarray(x))
    dy_in = torch.from_numpy(np.ascontiguousarray(y_backprop))
    x_n = _to_nchw_like_torch(x_in, data_format).to(torch.float64)
    dy_n = _to_nchw_like_torch(dy_in, data_format).to(torch.float64)
    axes = _channel_axes(x_n.dim())
    reduce_count = 1
    for axis in axes:
        reduce_count *= x_n.shape[axis]
    broadcast_shape = [1] * x_n.dim()
    broadcast_shape[1] = x_n.shape[1]

    scale_n = (
        torch.from_numpy(np.ascontiguousarray(scale))
        .to(torch.float64)
        .reshape(broadcast_shape)
    )
    mean_n = (
        torch.from_numpy(np.ascontiguousarray(reserve_space_1))
        .to(torch.float64)
        .reshape(broadcast_shape)
    )
    rs2 = torch.from_numpy(np.ascontiguousarray(reserve_space_2)).to(torch.float64)
    if is_training:
        rstd = rs2
    else:
        rstd = torch.rsqrt(rs2 + epsilon)
    rstd_n = rstd.reshape(broadcast_shape)

    x_hat = (x_n - mean_n) * rstd_n
    dscale = torch.sum(dy_n * x_hat, dim=axes)
    doffset = torch.sum(dy_n, dim=axes)

    if is_training:
        dx_n = (
            (
                dy_n
                - doffset.reshape(broadcast_shape) / reduce_count
                - x_hat * dscale.reshape(broadcast_shape) / reduce_count
            )
            * scale_n
            * rstd_n
        )
    else:
        dx_n = dy_n * scale_n * rstd_n
    dx = _from_nchw_like_torch(dx_n, data_format).to(x_in.dtype).numpy()
    empty = np.empty((0,), dtype=np.float32)
    return [
        dx,
        dscale.to(torch.float32).numpy(),
        doffset.to(torch.float32).numpy(),
        empty,
        empty,
    ]
