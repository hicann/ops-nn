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
"""LayerNormV3 references for Kernel/GEIR."""

import numpy as np
import torch

__golden__ = {"kernel": {"layer_norm_v3": "layer_norm_v3_golden"}}

__spec__ = {
    "layer_norm_v3": "LayerNormV3TestSpec",
}


def _check_axes_and_params(x, gamma, beta, begin_norm_axis, begin_params_axis):
    """校验轴范围和参数形状。"""
    for name, axis in (
        ("begin_norm_axis", begin_norm_axis),
        ("begin_params_axis", begin_params_axis),
    ):
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(f"{name} must be in [-x.ndim, x.ndim)")
    norm_axis = begin_norm_axis % x.ndim
    params_axis = begin_params_axis % x.ndim
    params_shape = tuple(x.shape[params_axis:])
    if tuple(gamma.shape) != params_shape or tuple(beta.shape) != params_shape:
        raise ValueError("gamma and beta must match x.shape[begin_params_axis:]")
    return tuple(range(norm_axis, x.ndim))


def layer_norm_v3_golden(
    x, gamma, beta, *, begin_norm_axis=0, begin_params_axis=0, epsilon=1e-5, **kwargs
):
    """FP64 计算，统计量 dtype 跟随 gamma。"""
    axes = _check_axes_and_params(x, gamma, beta, begin_norm_axis, begin_params_axis)
    x_fp64 = x.astype(np.float64)
    mean = np.mean(x_fp64, axis=axes, keepdims=True)
    variance = np.mean(np.square(x_fp64 - mean), axis=axes, keepdims=True)
    rstd = np.reciprocal(np.sqrt(variance + float(epsilon)))
    y = (x_fp64 - mean) * rstd * gamma.astype(np.float64) + beta.astype(np.float64)
    return [y.astype(x.dtype), mean.astype(gamma.dtype), rstd.astype(gamma.dtype)]


class LayerNormV3TorchThirdParty:
    """Torch 三方对照，低精度输入用 FP32 累积。"""

    def __call__(
        self,
        x,
        gamma,
        beta,
        begin_norm_axis=0,
        begin_params_axis=0,
        epsilon=1e-5,
        **kwargs,
    ):
        axes = _check_axes_and_params(
            x, gamma, beta, begin_norm_axis, begin_params_axis
        )
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_t = x.to(dtype)
        variance, mean = torch.var_mean(x_t, dim=axes, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(variance + float(epsilon))
        y = (x_t - mean) * rstd * gamma.to(dtype) + beta.to(dtype)
        return [y.to(x.dtype), mean.to(gamma.dtype), rstd.to(gamma.dtype)]


class LayerNormV3TestSpec:
    golden = staticmethod(layer_norm_v3_golden)
    third_party = {"torch": LayerNormV3TorchThirdParty}
