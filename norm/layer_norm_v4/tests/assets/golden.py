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
"""LayerNormV4 references for Kernel/GEIR, ACLNN and Torch E2E."""

import numpy as np
import torch

__golden__ = {"kernel": {"layer_norm_v4": "layer_norm_v4_golden"}}

__spec__ = {
    "layer_norm_v4": "LayerNormV4TestSpec",
    "aclnnLayerNorm": "AclnnLayerNormTestSpec",
    "torch.nn.functional.layer_norm": "TorchLayerNormTestSpec",
}


def _numpy_to_torch(array):
    """NumPy → CPU Tensor，兼容 BF16。"""
    if array is None:
        return None
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _layer_norm_golden_impl(x, normalized_shape, weight, bias, eps):
    """CPU 真值统一用 FP64 计算。"""
    return torch.ops.aten.native_layer_norm(
        x.to(torch.float64),
        normalized_shape,
        weight.to(torch.float64) if weight is not None else None,
        bias.to(torch.float64) if bias is not None else None,
        float(eps),
    )


def layer_norm_v4_golden(
    x, normalized_shape, gamma=None, beta=None, *, epsilon=1e-5, **kwargs
):
    """Kernel/GEIR：按 def.cpp 接收 NumPy 输入，返回 y、mean、rstd。"""
    if normalized_shape.ndim > 1:
        raise ValueError("normalized_shape must be a scalar or a 1D tensor")
    rank = normalized_shape.size
    if not 1 <= rank <= x.ndim:
        raise ValueError("normalized_shape length must be between 1 and x.ndim")
    ns = list(x.shape[-rank:])

    x_t = _numpy_to_torch(x)
    weight = None if gamma is None or gamma.size == 0 else _numpy_to_torch(gamma)
    bias = None if beta is None or beta.size == 0 else _numpy_to_torch(beta)
    outputs = _layer_norm_golden_impl(x_t, ns, weight, bias, epsilon)
    dtypes = (x.dtype, np.float32, np.float32)
    return [
        value.numpy().astype(dtype, copy=False) for value, dtype in zip(outputs, dtypes)
    ]


def aclnn_layer_norm_golden(
    input,
    normalizedShape,
    weightOptional=None,
    biasOptional=None,
    eps=1e-5,
    out=None,
    meanOutOptional=None,
    rstdOutOptional=None,
    **kwargs,
):
    """ACLNN：参数对应 aclnnLayerNormGetWorkspaceSize，不含 workspace/executor。"""
    outputs = _layer_norm_golden_impl(
        input, normalizedShape, weightOptional, biasOptional, eps
    )
    dtypes = (
        out.dtype if out is not None else input.dtype,
        meanOutOptional.dtype if meanOutOptional is not None else torch.float32,
        rstdOutOptional.dtype if rstdOutOptional is not None else torch.float32,
    )
    return [value.to(dtype) for value, dtype in zip(outputs, dtypes)]


def torch_layer_norm_golden(
    input, normalized_shape, weight=None, bias=None, eps=1e-5, **kwargs
):
    """PyTorch E2E：仅返回 y。"""
    y, _, _ = _layer_norm_golden_impl(input, normalized_shape, weight, bias, eps)
    return [y.to(input.dtype)]


class LayerNormV4TorchThirdParty:
    """Kernel 三方适配，保留输入 dtype 和 device。"""

    def __call__(
        self, x, normalized_shape, gamma=None, beta=None, epsilon=1e-5, **kwargs
    ):
        if normalized_shape.ndim > 1:
            raise ValueError("normalized_shape must be a scalar or a 1D tensor")
        rank = normalized_shape.numel()
        if not 1 <= rank <= x.ndim:
            raise ValueError("normalized_shape length must be between 1 and x.ndim")
        ns = list(x.shape[-rank:])
        gamma = None if gamma is None or gamma.numel() == 0 else gamma
        beta = None if beta is None or beta.numel() == 0 else beta
        y, mean, rstd = torch.ops.aten.native_layer_norm(
            x, ns, gamma, beta, float(epsilon)
        )
        return [y, mean.to(torch.float32), rstd.to(torch.float32)]


class AclnnLayerNormTorchThirdParty:
    """ACLNN 三方适配，仅转换输出 dtype。"""

    def __call__(
        self,
        input,
        normalizedShape,
        weightOptional=None,
        biasOptional=None,
        eps=1e-5,
        out=None,
        meanOutOptional=None,
        rstdOutOptional=None,
        **kwargs,
    ):
        outputs = torch.ops.aten.native_layer_norm(
            input, normalizedShape, weightOptional, biasOptional, float(eps)
        )
        dtypes = (
            out.dtype if out is not None else input.dtype,
            meanOutOptional.dtype if meanOutOptional is not None else torch.float32,
            rstdOutOptional.dtype if rstdOutOptional is not None else torch.float32,
        )
        return [value.to(dtype) for value, dtype in zip(outputs, dtypes)]


class LayerNormV4TestSpec:
    golden = staticmethod(layer_norm_v4_golden)
    third_party = {"torch": LayerNormV4TorchThirdParty}


class AclnnLayerNormTestSpec:
    golden = staticmethod(aclnn_layer_norm_golden)
    third_party = {"torch": AclnnLayerNormTorchThirdParty}


class TorchLayerNormTestSpec:
    golden = staticmethod(torch_layer_norm_golden)
    third_party = {"torch": "torch.nn.functional.layer_norm"}
