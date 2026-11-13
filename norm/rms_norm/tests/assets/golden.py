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

__golden__ = {
    "kernel": {"rms_norm": "rms_norm_golden"},
    "aclnn": {"aclnnRmsNorm": "aclnn_rms_norm_golden"},
    "e2e": {"torch_npu.npu_rms_norm": "aclnn_rms_norm_golden"},
}


# Kernel / GEIR 共用算子名；ACLNN / Torch 按实际 API 名注册。
__spec__ = {
    "rms_norm": "RmsNormSpec",
    "aclnnRmsNorm": "AclnnRmsNormSpec",
    "torch_npu.npu_rms_norm": "TorchRmsNormSpec",
}


def _rms_norm_reference(x, gamma, epsilon):
    """FP64 mathematical reference; also covers finite BF16/FP32 squared overflow."""
    import torch

    x_calc = x.to(torch.float64)
    gamma_calc = gamma.to(torch.float64)
    axes = tuple(range(x.dim() - gamma.dim(), x.dim()))
    variance = x_calc.square().mean(dim=axes, keepdim=True)
    rstd = torch.rsqrt(variance + epsilon)
    return (x_calc * rstd) * gamma_calc, rstd


def rms_norm_golden(x, gamma, *, epsilon=1e-6, **kwargs):
    """
    Golden function for rms_norm.
    All the parameters (names and order) follow @rms_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        y with the input dtype, and rstd with the calculation dtype.
    """
    # 归约 gamma 覆盖的末尾各维并保留维度。R 为空时 mean 产生 NaN，
    # A 为空时 rstd 也为空；不使用 reshape(..., -1)，避免零维歧义。
    # 保留 kernel 通路的 NumPy 惯例，在平方前提升精度以避免有限 FP32/BF16 溢出。
    x_calc = x.astype(np.float64)
    gamma_calc = gamma.astype(np.float64)
    axes = tuple(range(x.ndim - gamma.ndim, x.ndim))
    variance = np.mean(np.square(x_calc), axis=axes, keepdims=True)
    rstd = 1 / np.sqrt(variance + epsilon)
    y = (x_calc * rstd) * gamma_calc
    return y.astype(x.dtype, copy=False), rstd


def aclnn_rms_norm_golden(x, gamma, epsilon=1e-6, y_out=None, rstd_out=None, **kwargs):
    """
    Aclnn golden for aclnnRmsNorm.
    All the parameters (name & order) follow
        function `aclnnRmsNormGetWorkspaceSize` in aclnn_rms_norm.h
        without `workspaceSize` & `executor`.
    Inputs are torch.Tensor. Optional output placeholders are unused.
    The same entry also serves torch_npu.npu_rms_norm.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    y, rstd = _rms_norm_reference(x, gamma, epsilon)
    return y.to(x.dtype), rstd


class RmsNormXpuRef:
    """Actual Torch RMSNorm competitor; keep its FP32 arithmetic and overflow behavior.

    CPU golden uses FP64 independently, so common device overflow is not hidden.
    """

    def __call__(self, x, gamma, epsilon=1e-6, **kwargs):
        import torch

        calc_dtype = (
            torch.float64 if torch.float64 in (x.dtype, gamma.dtype) else torch.float32
        )
        x_calc = x.to(calc_dtype)
        y = torch.rms_norm(x_calc, tuple(gamma.shape), gamma.to(calc_dtype), epsilon)
        # 框架 RMSNorm 只返回 y，补充接口要求的 rstd。
        outer_shape = x.shape[: x.dim() - gamma.dim()]
        rstd = torch.rsqrt(
            x_calc.reshape((*outer_shape, gamma.numel()))
            .square()
            .mean(-1, keepdim=True)
            + epsilon
        ).reshape((*outer_shape, *([1] * gamma.dim())))
        return y.to(x.dtype), rstd


class TorchRmsNormXpuRef(RmsNormXpuRef):
    """Match the e2e API's ``input`` name for TTK's name-based XPU dispatch."""

    def __call__(self, input, gamma, epsilon=1e-6, **kwargs):
        return super().__call__(input, gamma, epsilon, **kwargs)


class RmsNormSpec:
    # 沿用 TTK 默认精度标准；GPU 参考由 xpu-server 在目标设备上实例化。
    golden = staticmethod(rms_norm_golden)
    third_party = {"torch": RmsNormXpuRef}


class AclnnRmsNormSpec(RmsNormSpec):
    golden = staticmethod(aclnn_rms_norm_golden)


class TorchRmsNormSpec(AclnnRmsNormSpec):
    third_party = {"torch": TorchRmsNormXpuRef}
