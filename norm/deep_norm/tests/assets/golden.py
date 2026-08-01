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
"""Torch CPU input and golden plugin for DeepNorm."""

import hashlib

import numpy as np
import torch
import ml_dtypes  # noqa: F401 - registers bfloat16 with NumPy


__golden__ = {
    "kernel": {
        "deep_norm": "deep_norm_golden",
        "DeepNorm": "deep_norm_golden",
    },
    "aclnn": {"aclnnDeepNorm": "deep_norm_aclnn_golden"},
}

__input__ = {
    "kernel": {
        "deep_norm": "deep_norm_input",
        "DeepNorm": "deep_norm_input",
    },
    "aclnn": {"aclnnDeepNorm": "deep_norm_aclnn_input"},
}


def _is_torch_tensor(value):
    return torch.is_tensor(value)


def _shape_of(value):
    return tuple(value.shape)


def _to_torch_fp64(value):
    if _is_torch_tensor(value):
        return value.detach().cpu().to(torch.float64)
    return torch.as_tensor(np.asarray(value).astype(np.float64), dtype=torch.float64)


def _numpy_dtype_from_name(dtype_name):
    if dtype_name in ("bfloat16", "bf16"):
        return ml_dtypes.bfloat16
    return np.dtype(dtype_name)


def _dtype_name(dtype):
    text = str(dtype)
    if text in ("torch.float32", "float32"):
        return "float32"
    if text in ("torch.float16", "float16"):
        return "float16"
    if text in ("torch.bfloat16", "bfloat16"):
        return "bfloat16"
    return np.dtype(dtype).name


def _torch_dtype_from_name(dtype_name):
    if dtype_name in ("float32", "fp32"):
        return torch.float32
    if dtype_name in ("float16", "fp16"):
        return torch.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _output_shapes(x, gamma):
    x_shape = _shape_of(x)
    gamma_ndim = len(_shape_of(gamma))
    mean_shape = x_shape[: len(x_shape) - gamma_ndim] + (1,) * gamma_ndim
    return mean_shape, mean_shape, x_shape


def _zero_numpy(shape, dtype_name):
    return np.zeros(shape, dtype=_numpy_dtype_from_name(dtype_name))


def _fallback_kernel_outputs(x, gamma, output_dtypes=None, output_shapes=None):
    if output_shapes:
        mean_shape, rstd_shape, y_shape = output_shapes
    else:
        mean_shape, rstd_shape, y_shape = _output_shapes(x, gamma)
    y_dtype = _dtype_name(np.asarray(x).dtype)
    if output_dtypes and len(output_dtypes) >= 3:
        y_dtype = output_dtypes[2]
    return (
        _zero_numpy(mean_shape, "float32"),
        _zero_numpy(rstd_shape, "float32"),
        _zero_numpy(y_shape, y_dtype),
    )


def _fallback_aclnn_outputs(mean_out, rstd_out, y_out):
    if _is_torch_tensor(mean_out):
        return (
            torch.zeros_like(mean_out),
            torch.zeros_like(rstd_out),
            torch.zeros_like(y_out),
        )
    return (
        np.zeros_like(mean_out),
        np.zeros_like(rstd_out),
        np.zeros_like(y_out),
    )


def _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon):
    x64 = _to_torch_fp64(x)
    gx64 = _to_torch_fp64(gx)
    beta64 = _to_torch_fp64(beta)
    gamma64 = _to_torch_fp64(gamma)
    gamma_ndim = gamma64.dim()
    if x64.dim() <= gamma_ndim or x64.numel() == 0 or gamma64.numel() == 0:
        raise ValueError("invalid or empty DeepNorm shape for golden computation")

    reduce_axes = tuple(range(x64.dim() - gamma_ndim, x64.dim()))
    hidden = x64 * float(alpha) + gx64
    mean64 = torch.mean(hidden, dim=reduce_axes, keepdim=True)
    centered = hidden - mean64
    var64 = torch.mean(centered * centered, dim=reduce_axes, keepdim=True)
    rstd64 = torch.rsqrt(var64 + float(epsilon))
    y64 = centered * rstd64 * gamma64 + beta64
    return mean64.to(torch.float32), rstd64.to(torch.float32), y64


def _torch_to_numpy(tensor, dtype_name):
    if dtype_name == "bfloat16":
        return (
            tensor.to(torch.bfloat16)
            .to(torch.float32)
            .numpy()
            .astype(ml_dtypes.bfloat16)
        )
    return (
        tensor.to(_torch_dtype_from_name(dtype_name))
        .numpy()
        .astype(_numpy_dtype_from_name(dtype_name), copy=False)
    )


def deep_norm_golden(
    x,
    gx,
    beta,
    gamma,
    alpha=0.3,
    epsilon=1e-6,
    output_dtypes=None,
    output_ori_shapes=None,
    output_shapes=None,
    **kwargs,
):
    """Return (mean, rstd, y) for TTK kernel mode."""
    try:
        mean, rstd, y = _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon)
    except ValueError:
        return _fallback_kernel_outputs(
            x, gamma, output_dtypes, output_ori_shapes or output_shapes
        )

    y_dtype = _dtype_name(np.asarray(x).dtype)
    if output_dtypes and len(output_dtypes) >= 3:
        y_dtype = _dtype_name(output_dtypes[2])
    return (
        mean.numpy(),
        rstd.numpy(),
        _torch_to_numpy(y, y_dtype),
    )


def deep_norm_aclnn_golden(
    x,
    gx,
    beta,
    gamma,
    alpha=0.3,
    epsilon=1e-6,
    mean_out=None,
    rstd_out=None,
    y_out=None,
    **kwargs,
):
    """Return (meanOut, rstdOut, yOut) for TTK ACLNN mode."""
    try:
        mean, rstd, y = _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon)
    except ValueError:
        return _fallback_aclnn_outputs(mean_out, rstd_out, y_out)

    if _is_torch_tensor(y_out):
        return mean.to(mean_out.dtype), rstd.to(rstd_out.dtype), y.to(y_out.dtype)
    return (
        mean.numpy().astype(np.asarray(mean_out).dtype, copy=False),
        rstd.numpy().astype(np.asarray(rstd_out).dtype, copy=False),
        _torch_to_numpy(y, _dtype_name(np.asarray(y_out).dtype)),
    )


def _set_first(value, target):
    if target is None:
        return
    if _is_torch_tensor(target):
        if target.numel() == 0:
            return
        index = (0,) * target.dim()
        target[index] = value
        return
    arr = np.asarray(target)
    if arr.size:
        arr.reshape(-1)[0] = value


def _fill_constant(value, target):
    if target is None:
        return
    if _is_torch_tensor(target):
        target.fill_(value)
        return
    np.asarray(target)[...] = value


def _stable_seed(testcase_name, input_index):
    key = f"{testcase_name}:{input_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _numeric_bounds(data_range):
    if not isinstance(data_range, (tuple, list)) or len(data_range) != 2:
        return None
    low, high = data_range
    if not isinstance(low, (int, float, np.number)) or not isinstance(
        high, (int, float, np.number)
    ):
        return None
    return float(low), float(high)


def _fill_stable_uniform(target, data_range, seed):
    if target is None:
        return
    bounds = _numeric_bounds(data_range)
    if bounds is None:
        return
    low, high = bounds
    shape = tuple(target.shape)
    rng = np.random.default_rng(seed)
    values = rng.uniform(low, high, size=shape).astype(np.float32)
    if _is_torch_tensor(target):
        source = torch.from_numpy(values).to(device=target.device, dtype=target.dtype)
        target.copy_(source)
        return
    arr = np.asarray(target)
    arr[...] = values.astype(arr.dtype, copy=False)


def _apply_stable_inputs(testcase_name, inputs, input_ranges):
    if not testcase_name or not input_ranges:
        return
    for index, target in enumerate(inputs):
        if index >= len(input_ranges):
            break
        _fill_stable_uniform(
            target, input_ranges[index], _stable_seed(testcase_name, index)
        )


def _apply_special_inputs(testcase_name, x, gx, beta, gamma):
    name = testcase_name or ""
    if "zero_var" in name:
        _fill_constant(1.0, x)
        _fill_constant(2.0, gx)
    if "inf_x" in name:
        _set_first(float("inf"), x)
    if "ninf_gx" in name:
        _set_first(float("-inf"), gx)
    if "nan_x" in name:
        _set_first(float("nan"), x)
    if "nan_gamma" in name:
        _set_first(float("nan"), gamma)
    if "nan_beta" in name:
        _set_first(float("nan"), beta)


def deep_norm_input(
    x,
    gx,
    beta,
    gamma,
    alpha=0.3,
    epsilon=1e-6,
    testcase_name=None,
    input_ranges=None,
    **kwargs,
):
    """Inject deterministic special values for kernel DFX cases."""
    _apply_stable_inputs(testcase_name, (x, gx, beta, gamma), input_ranges)
    _apply_special_inputs(testcase_name, x, gx, beta, gamma)
    return [x, gx, beta, gamma]


def deep_norm_aclnn_input(
    x,
    gx,
    beta,
    gamma,
    alpha=0.3,
    epsilon=1e-6,
    mean_out=None,
    rstd_out=None,
    y_out=None,
    testcase_name=None,
    input_ranges=None,
    **kwargs,
):
    """Inject deterministic special values for ACLNN DFX cases."""
    _apply_stable_inputs(testcase_name, (x, gx, beta, gamma), input_ranges)
    _apply_special_inputs(testcase_name, x, gx, beta, gamma)
    return [x, gx, beta, gamma]
