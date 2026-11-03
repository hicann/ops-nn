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
"""Torch CPU input and golden plugin for DeepNorm.

Enable mode uses FP32 intermediates and returns the declared output dtypes.
Promote mode keeps an FP64 reference for cross_check error ratios; the
third-party implementation uses FP32 intermediates and declared output dtypes.
Empty reductions produce NaN mean/rstd, while zero leading dimensions leave
all outputs empty, matching the arch35 empty-input kernel.
"""

import hashlib

import numpy as np
import torch


__spec__ = {
    "deep_norm": "DeepNormKernelSpec",
    "DeepNorm": "DeepNormKernelSpec",
    "aclnnDeepNorm": "DeepNormAclnnSpec",
}

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

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_ACLNN_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _is_torch_tensor(value):
    return torch.is_tensor(value)


def _to_torch(value):
    if _is_torch_tensor(value):
        return value.detach().cpu()
    array = np.asarray(value)
    if array.dtype.name == "bfloat16":
        # NumPy bf16 has no torch.from_numpy bridge. Conversion is exact.
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)


def _numpy_dtype_from_name(dtype_name):
    if dtype_name in ("bfloat16", "bf16"):
        # Only the NumPy BF16 bridge needs this optional CPU dependency.
        # Remote Torch references must also work without ml_dtypes installed.
        import ml_dtypes

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
    if text in ("torch.float64", "float64"):
        return "float64"
    return np.dtype(dtype).name


def _torch_dtype_from_name(dtype_name):
    if dtype_name in ("float32", "fp32"):
        return torch.float32
    if dtype_name in ("float16", "fp16"):
        return torch.float16
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_name in ("float64", "fp64"):
        return torch.float64
    raise ValueError(f"Unsupported DeepNorm output dtype: {dtype_name}")


def _validate_inputs(x, gx, beta, gamma):
    values = (x, gx, beta, gamma)
    if any(value is None for value in values):
        raise ValueError("DeepNorm requires x, gx, beta and gamma")
    dtype = _dtype_name(x.dtype)
    if dtype not in ("float16", "bfloat16", "float32", "float64"):
        raise ValueError(f"Unsupported DeepNorm input dtype: {dtype}")
    if any(_dtype_name(value.dtype) != dtype for value in values):
        raise ValueError("DeepNorm input dtypes must match")
    if not (2 <= len(x.shape) <= 8 and 1 <= len(gamma.shape) < len(x.shape)):
        raise ValueError("Invalid DeepNorm input ranks")
    if tuple(gx.shape) != tuple(x.shape) or tuple(beta.shape) != tuple(gamma.shape):
        raise ValueError("DeepNorm x/gx and beta/gamma shapes must match")
    if tuple(gamma.shape) != tuple(x.shape[-len(gamma.shape) :]):
        raise ValueError("DeepNorm gamma must match x trailing dimensions")


def _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon, promoted=False):
    _validate_inputs(x, gx, beta, gamma)
    operands = tuple(_to_torch(value) for value in (x, gx, beta, gamma))
    # Keep a high-precision anchor for cross_check, including when the
    # framework supplies promoted FP32 inputs for FP16/BF16 cases. Ordinary
    # execution and the third-party implementation retain FP32 intermediates.
    compute_dtype = torch.float64 if promoted else torch.float32
    operands = tuple(value.to(compute_dtype) for value in operands)
    x_work, gx_work, beta_work, gamma_work = operands
    gamma_ndim = gamma_work.dim()
    if x_work.dim() <= gamma_ndim:
        raise ValueError("x rank must exceed gamma rank")
    reduce_axes = tuple(range(x_work.dim() - gamma_ndim, x_work.dim()))
    hidden = x_work * alpha + gx_work
    mean = torch.mean(hidden, dim=reduce_axes, keepdim=True)
    centered = hidden - mean
    var = torch.mean(centered * centered, dim=reduce_axes, keepdim=True)
    rstd = torch.rsqrt(var + epsilon)
    y = centered * rstd * gamma_work + beta_work
    return mean, rstd, y


def _torch_to_numpy(tensor, dtype_name):
    if dtype_name == "bfloat16":
        return (
            tensor.to(torch.bfloat16)
            .to(torch.float32)
            .numpy()
            .astype(_numpy_dtype_from_name("bfloat16"))
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
    golden_mode="Enable",
    **kwargs,
):
    """Return (mean, rstd, y) for TTK kernel mode."""
    promoted = golden_mode == "Promote"
    mean, rstd, y = _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon, promoted)

    if promoted:
        return tuple(value.numpy() for value in (mean, rstd, y))
    y_dtype = _dtype_name(x.dtype)
    mean_dtype = "float32"
    rstd_dtype = "float32"
    if output_dtypes and len(output_dtypes) >= 3:
        y_dtype = _dtype_name(output_dtypes[2])
        mean_dtype = _dtype_name(output_dtypes[0])
        rstd_dtype = _dtype_name(output_dtypes[1])
    return (
        _torch_to_numpy(mean, mean_dtype),
        _torch_to_numpy(rstd, rstd_dtype),
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
    golden_mode=None,
    **kwargs,
):
    """Return (meanOut, rstdOut, yOut) for TTK ACLNN mode."""
    _validate_inputs(x, gx, beta, gamma)
    if any(value is None for value in (mean_out, rstd_out, y_out)):
        raise ValueError("DeepNorm ACLNN requires all three output slots")
    x_dtype, y_dtype = _dtype_name(x.dtype), _dtype_name(y_out.dtype)
    if golden_mode is not None:
        promoted = golden_mode == "Promote"
    else:
        # Compatibility with ACLNN TTK versions that omit golden_mode and keep
        # pure output slots at their declared dtype. Accept only exact supported
        # Promote pairs, rather than treating every mismatch as Promote.
        promote_pairs = {
            ("float32", "float16"),
            ("float32", "bfloat16"),
            ("float64", "float32"),
        }
        promoted = (x_dtype, y_dtype) in promote_pairs
        if not promoted and (x_dtype != y_dtype or x_dtype == "float64"):
            raise ValueError(
                "Cannot resolve DeepNorm ACLNN precision mode from dtype pair"
            )
    if not promoted and x_dtype == "float64":
        raise ValueError("FP64 input is supported only as a promoted CPU reference")
    mean, rstd, y = _compute_deep_norm(x, gx, beta, gamma, alpha, epsilon, promoted)

    if promoted:
        return (
            (mean, rstd, y)
            if _is_torch_tensor(x)
            else tuple(value.numpy() for value in (mean, rstd, y))
        )
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


class _DeepNormCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, alpha=0.3, epsilon=1e-6, **kwargs):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def __call__(self, x, gx, beta, gamma, **kwargs):
        _validate_inputs(x, gx, beta, gamma)
        out_dtype = x.dtype
        x_work = x.to(torch.float32)
        gx_work = gx.to(torch.float32)
        beta_work = beta.to(torch.float32)
        gamma_work = gamma.to(torch.float32)
        gamma_ndim = gamma_work.dim()
        reduce_axes = tuple(range(x_work.dim() - gamma_ndim, x_work.dim()))
        hidden = x_work * self.alpha + gx_work
        mean = torch.mean(hidden, dim=reduce_axes, keepdim=True)
        centered = hidden - mean
        var = torch.mean(centered * centered, dim=reduce_axes, keepdim=True)
        rstd = torch.rsqrt(var + self.epsilon)
        y = centered * rstd * gamma_work + beta_work
        return [mean.to(torch.float32), rstd.to(torch.float32), y.to(out_dtype)]


class DeepNormKernelSpec:
    golden = deep_norm_golden
    input = deep_norm_input
    third_party = {"torch": _DeepNormCompose}
    tolerance = _KERNEL_TOLERANCE


class DeepNormAclnnSpec:
    golden = deep_norm_aclnn_golden
    input = deep_norm_aclnn_input
    third_party = {"torch": _DeepNormCompose}
    tolerance = _ACLNN_TOLERANCE


# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 aclnnDeepNorm.
