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
"""Golden plugin for INInferV2 (instance normalization inference).

    y              = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta   # gamma/beta 可选
                     无 gamma/beta 时 y = (x - mean) / sqrt(variance + epsilon)
    batch_mean     = mean          （透传拷贝）
    batch_variance = variance      （透传拷贝）

与 kernel 一致：全部按 fp32 计算（fp16 的 x 先升 fp32），y 单次舍入写回输入 dtype；
scale = gamma / sqrt(var + eps) 先算（对齐 910b high_performance 语义）。

格式说明：算子 ND-only（tiling 仅接受 ND/NCHW 标签），C 恒在 dim1；
_is_channels_last 仅为防御性保留。
"""

import numpy as np

__golden__ = {
    "kernel": {"in_infer_v2": "in_infer_v2_golden"},
    "aclnn": {"aclnnINInferV2": "aclnn_in_infer_v2_golden"},
}


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):  # torch tensor
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _is_channels_last(x_shape, c):
    """判断 C 是否在末维（NHWC）；C 在 dim1（ND/NCHW）时返回 False。

    用例 shape 设计保证 d1 与末维只有一个等于 C，无歧义。
    """
    if len(x_shape) < 3:
        return False
    return x_shape[1] != c and x_shape[-1] == c


def _compute(x, gamma, beta, mean, variance, epsilon):
    """核心计算：返回 (y fp32计算后按输入 dtype 舍入, mean, variance)。"""
    out_dtype = _to_numpy(x).dtype
    x_np = _to_numpy(x).astype(np.float32)
    mean_np = _to_numpy(mean).astype(np.float32)
    var_np = _to_numpy(variance).astype(np.float32)
    n, c = mean_np.shape[0], mean_np.shape[1]

    if _is_channels_last(x_np.shape, c):
        bcast_shape = [n] + [1] * (x_np.ndim - 2) + [c]
    else:
        bcast_shape = [n, c] + [1] * (x_np.ndim - 2)
    mean_b = mean_np.reshape(bcast_shape)
    var_b = var_np.reshape(bcast_shape)

    sqrt_var = np.sqrt(var_b + np.float32(epsilon))
    gamma_np = _to_numpy(gamma)
    if gamma_np is not None:
        gamma_np = gamma_np.astype(np.float32)
        beta_np = _to_numpy(beta).astype(np.float32)
        scale = (gamma_np / sqrt_var.reshape(gamma_np.shape)).reshape(bcast_shape)
        beta_b = beta_np.reshape(bcast_shape)
        y = (x_np - mean_b) * scale + beta_b
    else:
        y = (x_np - mean_b) / sqrt_var
    return y.astype(out_dtype), mean_np, var_np


def in_infer_v2_golden(x, gamma, beta, mean, variance, epsilon=1e-5, **kwargs):
    """Golden for in_infer_v2 kernel 模式。参数顺序同 def（不含输出）。"""
    del kwargs
    y, mean_np, var_np = _compute(x, gamma, beta, mean, variance, epsilon)
    return [y, mean_np.copy(), var_np.copy()]


def aclnn_in_infer_v2_golden(
    x,
    gamma,
    beta,
    mean,
    variance,
    epsilon=1e-5,
    y=None,
    batch_mean=None,
    batch_variance=None,
    *args,
    **kwargs,
):
    """Golden for aclnnINInferV2（ttk aclnn 模式）。

    注意：ttk aclnn 模式按 C 头文件参数序位置传参（AclnnParamPlan.build_args）——
    epsilon 在 outputs 之前，outputs 之后还有 workspaceSize/executor 占位，
    故签名必须是 C 序 + *args 吞掉尾部占位，不能按 csv tensor 序。
    batch_mean/batch_variance 为 None（可选输出缺席）时仅返回 [y]。
    """
    del y, args, kwargs
    has_batch_out = batch_mean is not None or batch_variance is not None
    y_np, mean_np, var_np = _compute(x, gamma, beta, mean, variance, epsilon)
    if not has_batch_out:
        return [y_np]
    return [y_np, mean_np.copy(), var_np.copy()]
