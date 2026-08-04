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
"""Golden plugin for INInferV2 (instance normalization inference)，torch 竞品算子拼接实现。

    y              = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta   # gamma/beta 可选
                     无 gamma/beta 时 y = (x - mean) / sqrt(variance + epsilon)
    batch_mean     = mean          （透传拷贝）
    batch_variance = variance      （透传拷贝）

与 kernel 一致：全部按 float32 计算（fp16 的 x 先升 fp32），y 单次舍入写回输入 dtype；
scale = gamma / sqrt(var + eps) 先算（对齐 910b high_performance 语义）。

格式说明：算子 ND-only（tiling 仅接受 ND/NCHW 标签），C 恒在 dim1；
_is_channels_last 仅为防御性保留。
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {"in_infer_v2": "in_infer_v2_golden"},
    "aclnn": {"aclnnINInferV2": "aclnn_in_infer_v2_golden"},
}


def _to_torch_f32(tensor):
    """输入归一为 torch float32（接受 numpy / torch tensor，None 透传；ml_dtypes.bfloat16 等 numpy 扩展 dtype 先升 fp32）。"""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().to(torch.float32)
    arr = np.asarray(tensor)
    if arr.dtype not in (
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int16,
        np.int8,
        np.uint8,
    ):
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(torch.float32)


def _out_dtype(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().dtype
    return np.asarray(x).dtype


def _is_channels_last(x_shape, c):
    """判断 C 是否在末维（NHWC）；C 在 dim1（ND/NCHW）时返回 False。

    用例 shape 设计保证 d1 与末维只有一个等于 C，无歧义。
    """
    if len(x_shape) < 3:
        return False
    return x_shape[1] != c and x_shape[-1] == c


def _compute(x, gamma, beta, mean, variance, epsilon):
    """核心计算：返回 (y fp32计算后按输入 dtype 舍入, mean, variance)。"""
    out_dtype = _out_dtype(x)
    x_t = _to_torch_f32(x)
    mean_t = _to_torch_f32(mean)
    var_t = _to_torch_f32(variance)
    n, c = mean_t.shape[0], mean_t.shape[1]

    if _is_channels_last(tuple(x_t.shape), c):
        bcast_shape = [n] + [1] * (x_t.dim() - 2) + [c]
    else:
        bcast_shape = [n, c] + [1] * (x_t.dim() - 2)
    mean_b = mean_t.reshape(bcast_shape)
    var_b = var_t.reshape(bcast_shape)

    sqrt_var = torch.sqrt(var_b + float(np.float32(epsilon)))
    gamma_t = _to_torch_f32(gamma)
    if gamma_t is not None:
        beta_t = _to_torch_f32(beta)
        scale = (gamma_t / sqrt_var.reshape(gamma_t.shape)).reshape(bcast_shape)
        beta_b = beta_t.reshape(bcast_shape)
        y = (x_t - mean_b) * scale + beta_b
    else:
        y = (x_t - mean_b) / sqrt_var
    return y.numpy().astype(out_dtype), mean_t.numpy(), var_t.numpy()


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
