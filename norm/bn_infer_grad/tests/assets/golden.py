#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {
        "bn_infer_grad": "bn_infer_grad_golden"
    }
}


def _expand_inv_std(inv_std, grads_shape, format_mode):
    '''
    将 [C] 形状的 inv_std 按照 format_mode 广播到 grads 的整张布局。

    format_mode:
      0 -> NCHW (4D)
      1 -> NHWC (4D)
      2 -> NC1HWC0 (5D, channel = C1*C0)
    '''
    fmt = int(format_mode)
    if fmt == 0:
        # NCHW: (N, C, H, W)
        return inv_std.reshape(1, -1, 1, 1)
    if fmt == 1:
        # NHWC: (N, H, W, C)
        return inv_std.reshape(1, 1, 1, -1)
    if fmt == 2:
        # NC1HWC0: (N, C1, H, W, C0)
        c1 = grads_shape[1]
        c0 = grads_shape[-1]
        return inv_std.reshape(1, c1, 1, 1, c0)
    raise ValueError(f"unsupported format_mode={format_mode}")


def bn_infer_grad_golden(grads, scale, batch_variance, **kwargs):
    '''
    Golden function for bn_infer_grad.

    x_backprop = grads * scale * 1/sqrt(batch_variance + epsilon)

    Args:
        grads: 上游梯度 (numpy.ndarray, fp16/fp32/bf16)
        scale: 通道缩放 gamma (numpy.ndarray, fp32, shape=[C])
        batch_variance: 推理阶段方差 (numpy.ndarray, fp32, shape=[C])

    kwargs 支持:
        epsilon: float，默认 0.0001
        format_mode: int，0=NCHW(默认) / 1=NHWC / 2=NC1HWC0

    Returns:
        x_backprop (numpy.ndarray)，shape 与 grads 一致，dtype 与 grads 一致
    '''
    epsilon = float(kwargs.get("epsilon", 1e-4))
    format_mode = int(kwargs.get("format_mode", 0))

    grads_dtype = grads.dtype
    grads_fp32 = grads.astype(np.float32, copy=False)
    scale_fp32 = scale.astype(np.float32, copy=False).reshape(-1)
    var_fp32 = batch_variance.astype(np.float32, copy=False).reshape(-1)

    inv_std = scale_fp32 / np.sqrt(var_fp32 + epsilon)
    inv_std_b = _expand_inv_std(inv_std, grads.shape, format_mode)

    result = grads_fp32 * inv_std_b
    if grads_dtype.name in ("bfloat16", "float16"):
        result = result.astype(grads_dtype, copy=False)
    else:
        result = result.astype(grads_dtype, copy=False)
    return result
