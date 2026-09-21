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
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
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
    if tensor.ndim != 1 or tensor.numel() != channel_len:
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
    import tensorflow as tf

    data_format = _infer_x_format(
        _to_torch(x), _to_torch(scale), kwargs.get("input_formats")
    )
    channel_axis = -1 if data_format in ("NHWC", "NDHWC") else 1
    channel_len = np.asarray(x).shape[channel_axis]
    for name, value in (
        ("scale", scale),
        ("offset", offset),
        ("mean", mean),
        ("variance", variance),
    ):
        array = np.asarray(value)
        if array.ndim != 1 or array.size != channel_len:
            raise ValueError(
                f"BNInfer {name} must be 1-D with {channel_len} elements; "
                f"got shape {array.shape}"
            )

    x_tensor = tf.convert_to_tensor(x)
    scale_tensor = tf.reshape(tf.convert_to_tensor(scale), [-1])
    offset_tensor = tf.reshape(tf.convert_to_tensor(offset), [-1])
    mean_tensor = tf.reshape(tf.convert_to_tensor(mean), [-1])
    variance_tensor = tf.reshape(tf.convert_to_tensor(variance), [-1])
    tensors = (x_tensor, scale_tensor, offset_tensor, mean_tensor, variance_tensor)
    compute_dtype = max(
        (tensor.dtype for tensor in tensors), key=lambda dtype: dtype.size
    )
    # TTK Promote advances each input from its original dtype independently.
    # Align mixed promoted inputs upward to the highest resulting dtype so TF
    # arithmetic is type-consistent; this does not select or narrow precision.
    x_tensor, scale_tensor, offset_tensor, mean_tensor, variance_tensor = (
        tf.cast(tensor, compute_dtype) for tensor in tensors
    )
    if data_format == "NHWC":
        x_tensor = tf.transpose(x_tensor, [0, 3, 1, 2])
    elif data_format == "NDHWC":
        x_tensor = tf.transpose(x_tensor, [0, 4, 1, 2, 3])
    shape = [1, -1] + [1] * (len(x_tensor.shape) - 2)
    y = (x_tensor - tf.reshape(mean_tensor, shape)) * tf.math.rsqrt(
        tf.reshape(variance_tensor, shape) + epsilon
    )
    y = y * tf.reshape(scale_tensor, shape) + tf.reshape(offset_tensor, shape)
    if data_format == "NHWC":
        y = tf.transpose(y, [0, 2, 3, 1])
    elif data_format == "NDHWC":
        y = tf.transpose(y, [0, 2, 3, 4, 1])
    return [y.numpy()]


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
        output_dtype = x_tensor.dtype
        data_format = _infer_x_format(x_tensor, scale, merged.get("input_formats"))
        if x_tensor.dtype in (torch.float16, torch.bfloat16):
            x_tensor = x_tensor.to(torch.float32)
        if data_format == "NHWC":
            x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous()
            inverse_perm = (0, 2, 3, 1)
        elif data_format == "NDHWC":
            x_tensor = x_tensor.permute(0, 4, 1, 2, 3).contiguous()
            inverse_perm = (0, 2, 3, 4, 1)
        else:
            inverse_perm = None
        channel_len = x_tensor.shape[1]

        def strict_param(value, name):
            tensor = value if isinstance(value, torch.Tensor) else _to_torch(value)
            if tensor.ndim != 1 or tensor.numel() != channel_len:
                raise ValueError(
                    f"BNInfer {name} must be 1-D with {channel_len} elements; "
                    f"got shape {tuple(tensor.shape)}"
                )
            return tensor.to(device=x_tensor.device, dtype=x_tensor.dtype).contiguous()

        mean_tensor = strict_param(mean, "mean")
        variance_tensor = strict_param(variance, "variance")
        scale_tensor = strict_param(scale, "scale")
        offset_tensor = strict_param(offset, "offset")
        shape = [1, -1] + [1] * (x_tensor.ndim - 2)

        # Match the Ascend 950 kernel's explicit FP32 arithmetic order. Keep
        # these as separate Torch operations so no fused batch-norm backend can
        # fold scale and reciprocal standard deviation into a different order.
        y = torch.sub(x_tensor, mean_tensor.reshape(shape))
        y = torch.mul(y, scale_tensor.reshape(shape))
        rstd = torch.rsqrt(variance_tensor.reshape(shape) + call_epsilon)
        y = torch.mul(y, rstd)
        y = torch.add(y, offset_tensor.reshape(shape))
        if inverse_perm is not None:
            y = y.permute(*inverse_perm).contiguous()
        return [y.to(output_dtype)]


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
# - ACLNN/Torch E2E: BNInfer has no public aclnnBNInfer interface or dedicated
#   torch_npu binding.
# - TensorFlow E2E: framework/bn_infer_tf_plugin.cpp provides TensorFlow parser
#   connectivity, but TensorFlow 2.16.1 has no tf.raw_ops.BNInfer callable.
#   Parser connectivity is therefore validated separately and is not an E2E
#   TestSpec registration.
# - ONNX/Caffe: no parser plugin is delivered under this operator directory.
