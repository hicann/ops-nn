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

import torch
import torch.nn.functional as F
import numpy as np

try:
    import ml_dtypes
except ImportError:  # pragma: no cover
    ml_dtypes = None


__spec__ = {
    "bn_infer": "BNInferKernelSpec",
}

_TOL = {
    "float32": {"standard": "stat_rel_err", "threshold": 1e-2},
    "float16": {"standard": "stat_rel_err", "threshold": 2e-2},
    "bfloat16": {"standard": "stat_rel_err", "threshold": 4e-2},
}


def _dtype_name(dtype):
    return getattr(dtype, "name", str(dtype)).lower()


def _torch_dtype(dtype_name):
    if dtype_name in ("float", "float32"):
        return torch.float32
    if dtype_name in ("float16", "half"):
        return torch.float16
    if dtype_name in ("bfloat16",):
        return torch.bfloat16
    if dtype_name in ("float64", "double"):
        return torch.float64
    return None


def _numpy_dtype(dtype_name):
    if dtype_name in ("bfloat16",) and ml_dtypes is not None:
        return ml_dtypes.bfloat16
    if dtype_name in ("float", "float32"):
        return np.float32
    if dtype_name in ("float16", "half"):
        return np.float16
    if dtype_name in ("float64", "double"):
        return np.float64
    return None


def _output_dtype(kwargs, fallback):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes:
        dtype_name = str(output_dtypes[0]).lower()
        torch_dtype = _torch_dtype(dtype_name)
        numpy_dtype = _numpy_dtype(dtype_name)
        if torch_dtype is not None and numpy_dtype is not None:
            return torch_dtype, numpy_dtype
    fallback_name = _dtype_name(np.asarray(fallback).dtype)
    return _torch_dtype(fallback_name), _numpy_dtype(fallback_name)


def _call_kwargs(kwargs):
    call_kwargs = dict(kwargs)
    call_kwargs.pop("epsilon", None)
    call_kwargs.pop("output_dtype", None)
    return call_kwargs


def _to_torch(array):
    dtype_name = _dtype_name(np.asarray(array).dtype)
    if dtype_name == "bfloat16":
        return torch.from_numpy(np.asarray(array, dtype=np.float32))
    return torch.as_tensor(array)


def _f32_floor(tensor):
    return (
        tensor.to(torch.float32)
        if tensor.dtype in (torch.float16, torch.bfloat16)
        else tensor
    )


def _prepare_param(tensor, channel_len, dtype, device):
    tensor = _f32_floor(tensor)
    if tensor.ndim == 1 and tensor.numel() == channel_len:
        param = tensor.to(device=device, dtype=dtype).reshape(-1)
        return param.contiguous()
    if (
        tensor.ndim > 1
        and tensor.shape[-1] == channel_len
        and tensor.numel() % channel_len == 0
    ):
        # TTK 的通用探测偶尔会把参数张量喂成批量占位形状；golden 只取首个通道向量，
        # 保证第三方探测可运行，同时不放松正式一维参数契约。
        tensor = tensor.reshape(-1, channel_len)[0]
    elif tensor.numel() != channel_len:
        raise ValueError(
            f"BNInfer parameter must be 1-D with {channel_len} elements; "
            f"got shape {tuple(tensor.shape)}"
        )
    param = tensor.to(device=device, dtype=dtype).reshape(-1)
    return param.contiguous()


def _channel_first(x, data_format):
    fmt = str(data_format).upper()
    if fmt == "NHWC":
        return x.permute(0, 3, 1, 2), (0, 2, 3, 1)
    if fmt == "NDHWC":
        return x.permute(0, 4, 1, 2, 3), (0, 2, 3, 4, 1)
    return x, None


def _infer_x_format(x, scale, input_formats):
    if input_formats:
        return (
            input_formats[0]
            if isinstance(input_formats, (list, tuple))
            else input_formats
        )

    # TTK's remote third-party provider sends tensors and attributes only. The
    # three-way case set keeps the channel dimension unambiguous for rank 4/5.
    if (
        x.ndim in (4, 5)
        and x.shape[-1] == scale.numel()
        and x.shape[1] != scale.numel()
    ):
        return "NHWC" if x.ndim == 4 else "NDHWC"
    return "ND"


def _compute(
    x,
    scale,
    offset,
    mean,
    variance,
    epsilon=1e-5,
    output_dtype=None,
    promote_to_fp64=False,
    **kwargs,
):
    """Compute BNInfer through PyTorch batch_norm in inference mode."""
    x_format = _infer_x_format(x, scale, kwargs.get("input_formats"))

    x_compute = x.to(torch.float64) if promote_to_fp64 else _f32_floor(x)
    x_cf, inverse_perm = _channel_first(x_compute, x_format)
    channel_len = x_cf.shape[1]
    compute_dtype = x_cf.dtype
    device = x_cf.device
    weight = _prepare_param(scale, channel_len, compute_dtype, device)
    bias = _prepare_param(offset, channel_len, compute_dtype, device)
    running_mean = _prepare_param(mean, channel_len, compute_dtype, device)
    running_var = _prepare_param(variance, channel_len, compute_dtype, device)
    y = F.batch_norm(
        x_cf,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
        momentum=0.1,
        eps=float(epsilon),
    )

    if inverse_perm is not None:
        y = y.permute(*inverse_perm).contiguous()
    if output_dtype is not None:
        y = y.to(output_dtype)
    return [y]


def _compute_numpy(x, scale, offset, mean, variance, epsilon=1e-5, **kwargs):
    call_kwargs = _call_kwargs(kwargs)
    result = _compute(
        _to_torch(x),
        _to_torch(scale),
        _to_torch(offset),
        _to_torch(mean),
        _to_torch(variance),
        epsilon=epsilon,
        output_dtype=torch.float64,
        promote_to_fp64=True,
        **call_kwargs,
    )
    y_tensor = result[0].detach().cpu()
    return [y_tensor.numpy()]


class _BNInferCompose:
    """Third-party baseline executed by the TTK remote provider."""

    def __init__(self, epsilon=1e-5, **kwargs):
        self.epsilon = float(epsilon)
        self.kwargs = kwargs

    def __call__(self, x, scale, offset, mean, variance, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        call_epsilon = float(merged.pop("epsilon", self.epsilon))
        merged.pop("output_dtype", None)
        for input_name in ("x", "scale", "offset", "mean", "variance"):
            merged.pop(input_name, None)
        x_tensor = x if isinstance(x, torch.Tensor) else _to_torch(x)
        return _compute(
            x_tensor,
            scale if isinstance(scale, torch.Tensor) else _to_torch(scale),
            offset if isinstance(offset, torch.Tensor) else _to_torch(offset),
            mean if isinstance(mean, torch.Tensor) else _to_torch(mean),
            variance if isinstance(variance, torch.Tensor) else _to_torch(variance),
            epsilon=call_epsilon,
            output_dtype=x_tensor.dtype,
            **merged,
        )


class BNInferKernelSpec:
    """Kernel and GEIR spec backed by the PyTorch competitor interface."""

    @staticmethod
    def golden(x, scale, offset, mean, variance, epsilon=1e-5, **kwargs):
        return _compute_numpy(
            x, scale, offset, mean, variance, epsilon=epsilon, **kwargs
        )

    third_party = {"torch": _BNInferCompose}
    tolerance = _TOL


def __golden_bn_infer(x, scale, offset, mean, variance, epsilon=1e-5, **kwargs):
    return BNInferKernelSpec.golden(
        x, scale, offset, mean, variance, epsilon=epsilon, **kwargs
    )


__golden__ = {"kernel": {"bn_infer": "__golden_bn_infer"}}


# Not registered in __spec__:
# - aclnn: BNInfer has no public aclnnBNInfer interface in norm/bn_infer.
# - e2e: torch_npu has no dedicated BNInfer torch binding for this GE internal op.
