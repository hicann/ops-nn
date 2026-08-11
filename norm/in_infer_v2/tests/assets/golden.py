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

"""INInferV2 kernel/GEIR golden in the TestSpec multi-path format.

For an ND input laid out as ``[N, C, R...]`` the operator computes::

    y = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta

When ``gamma`` and ``beta`` are both absent, the scale/add branch is omitted and
``y = (x - mean) / sqrt(variance + epsilon)``.  ``batch_mean`` and
``batch_variance`` are copies of the corresponding inputs.

The CPU true-value path is a Torch competitor composition.  It lifts fp16 to at
least fp32 and preserves fp64 inputs supplied by TTK Promote; promoted values are
never narrowed.  The independent third-party composition mirrors the arch35
kernel's float32 arithmetic and operation order before casting ``y`` back to the
input dtype.
"""

import numpy as np
import torch


# Kernel and GEIR resolve the same snake-case operator key and share one Spec.
__spec__ = {"in_infer_v2": "INInferV2KernelSpec"}

# Compatibility entry for the historical kernel golden loader.
__golden__ = {
    "kernel": {"in_infer_v2": "in_infer_v2_golden"},
}


_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """Read a scalar attribute, including the legacy nested-attributes form."""
    value = kwargs.get(name)
    if value is None and isinstance(kwargs.get("attributes"), dict):
        value = kwargs["attributes"].get(name)
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default
    return value


def _resolve_epsilon(epsilon, kwargs):
    values = dict(kwargs)
    values.setdefault("epsilon", epsilon)
    return float(_attr(values, "epsilon", 1e-5))


def _as_tensor(value):
    """Convert a Kernel/GEIR NumPy input to a CPU Torch tensor losslessly."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(value)))


def _reference_dtype(*tensors):
    """Select at least fp32 while retaining any wider promoted float dtype."""
    dtype = torch.float32
    for tensor in tensors:
        if tensor is not None and tensor.dtype.is_floating_point:
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _stat_matrix(tensor, n, c, name):
    expected = n * c
    if tensor.numel() != expected:
        raise ValueError(
            f"{name} must contain N*C={expected} elements, got {tensor.numel()}"
        )
    return torch.reshape(tensor, (n, c))


def _compute(x, gamma, beta, mean, variance, epsilon):
    """Sole Torch true-value core; return outputs in def.cpp order."""
    if mean is None or variance is None:
        raise ValueError("mean and variance are required by INInferV2 tiling")
    if (gamma is None) != (beta is None):
        raise ValueError("gamma and beta must be both present or both absent")

    x_tensor = _as_tensor(x)
    gamma_tensor = _as_tensor(gamma) if gamma is not None else None
    beta_tensor = _as_tensor(beta) if beta is not None else None
    mean_tensor = _as_tensor(mean)
    variance_tensor = _as_tensor(variance)

    if x_tensor.ndim < 2:
        raise ValueError(f"x rank must be at least 2, got {x_tensor.ndim}")
    n, c = x_tensor.shape[:2]
    compute_dtype = _reference_dtype(
        x_tensor, gamma_tensor, beta_tensor, mean_tensor, variance_tensor
    )
    x_compute = x_tensor.to(dtype=compute_dtype)
    mean_compute = mean_tensor.to(dtype=compute_dtype)
    variance_compute = variance_tensor.to(dtype=compute_dtype)

    mean_matrix = _stat_matrix(mean_compute, n, c, "mean")
    variance_matrix = _stat_matrix(variance_compute, n, c, "variance")
    # The host reads a Float attribute, so preserve that fp32 quantization even
    # when Promote has lifted tensor inputs to fp64.
    epsilon_f32 = torch.tensor(float(epsilon), dtype=torch.float32)
    epsilon_tensor = epsilon_f32.to(dtype=compute_dtype)
    inverse_std = torch.rsqrt(torch.add(variance_matrix, epsilon_tensor))

    broadcast_shape = (n, c) + (1,) * (x_tensor.ndim - 2)
    mean_broadcast = torch.reshape(mean_matrix, broadcast_shape)
    centered = torch.sub(x_compute, mean_broadcast)
    normalized = torch.mul(centered, torch.reshape(inverse_std, broadcast_shape))

    if gamma_tensor is None:
        y = normalized
    else:
        gamma_compute = gamma_tensor.to(dtype=compute_dtype)
        beta_compute = beta_tensor.to(dtype=compute_dtype)
        gamma_matrix = _stat_matrix(gamma_compute, n, c, "gamma")
        beta_matrix = _stat_matrix(beta_compute, n, c, "beta")
        scaled = torch.mul(normalized, torch.reshape(gamma_matrix, broadcast_shape))
        y = torch.add(scaled, torch.reshape(beta_matrix, broadcast_shape))

    # InferShape keeps the original mean/variance shapes for these two outputs.
    return [y, torch.clone(mean_compute), torch.clone(variance_compute)]


def _normalize_dtype_name(dtype):
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return {
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }.get(name, name)


def _input_dtype_name(value):
    if isinstance(value, torch.Tensor):
        return _normalize_dtype_name(value.dtype)
    return np.asarray(value).dtype.name


def _numpy_outputs(outputs, output_dtypes):
    dtype_names = [_normalize_dtype_name(dtype) for dtype in (output_dtypes or ())]
    result = []
    for index, output in enumerate(outputs):
        array = output.detach().cpu().contiguous().numpy()
        if index < len(dtype_names) and dtype_names[index] is not None:
            array = array.astype(dtype_names[index], copy=False)
        result.append(np.ascontiguousarray(array))
    return result


def _kernel_golden(
    x,
    gamma,
    beta,
    mean,
    variance,
    epsilon=1e-5,
    **kwargs,
):
    """Kernel/GEIR adapter: NumPy inputs and a NumPy output list."""
    epsilon_value = _resolve_epsilon(epsilon, kwargs)
    outputs = _compute(x, gamma, beta, mean, variance, epsilon_value)
    output_dtypes = kwargs.get("output_dtypes")
    if not output_dtypes:
        output_dtypes = (
            _input_dtype_name(x),
            _input_dtype_name(mean),
            _input_dtype_name(variance),
        )
    return _numpy_outputs(outputs, output_dtypes)


class _INInferV2Compose:
    """Independent Torch composition matching the arch35 device arithmetic."""

    def __init__(self, epsilon=1e-5, **kwargs):
        epsilon_value = _resolve_epsilon(epsilon, kwargs)
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(epsilon_value, dtype=torch.float32).item())

    def __call__(self, x, gamma, beta, mean, variance, **kwargs):
        del kwargs
        if mean is None or variance is None:
            raise ValueError("mean and variance are required by INInferV2 tiling")
        if (gamma is None) != (beta is None):
            raise ValueError("gamma and beta must be both present or both absent")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(f"INInferV2 supports only float16/float32 x, got {x.dtype}")

        n, c = x.shape[:2]
        broadcast_shape = (n, c) + (1,) * (x.ndim - 2)

        # INInferV2Kernel::Process converts every arithmetic operand to fp32.
        x_f32 = x.to(dtype=torch.float32)
        mean_f32 = _stat_matrix(mean.to(dtype=torch.float32), n, c, "mean")
        variance_f32 = _stat_matrix(variance.to(dtype=torch.float32), n, c, "variance")
        epsilon_f32 = variance_f32.new_tensor(self.epsilon)
        sqrt_variance = torch.sqrt(torch.add(variance_f32, epsilon_f32))
        centered = torch.sub(x_f32, torch.reshape(mean_f32, broadcast_shape))

        if gamma is None:
            y_f32 = torch.div(centered, torch.reshape(sqrt_variance, broadcast_shape))
        else:
            gamma_f32 = _stat_matrix(gamma.to(dtype=torch.float32), n, c, "gamma")
            beta_f32 = _stat_matrix(beta.to(dtype=torch.float32), n, c, "beta")
            scale = torch.div(gamma_f32, sqrt_variance)
            scaled = torch.mul(centered, torch.reshape(scale, broadcast_shape))
            y_f32 = torch.add(scaled, torch.reshape(beta_f32, broadcast_shape))

        # The first output is stored in x dtype; the two copy outputs are fp32.
        return [
            y_f32.to(dtype=x.dtype),
            mean.to(dtype=torch.float32).clone(),
            variance.to(dtype=torch.float32).clone(),
        ]


class INInferV2KernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow in_infer_v2_def.cpp."""

    golden = _kernel_golden
    third_party = {"torch": _INInferV2Compose}
    tolerance = _TOL


def in_infer_v2_golden(
    x,
    gamma,
    beta,
    mean,
    variance,
    epsilon=1e-5,
    *args,
    **kwargs,
):
    """Compatibility ``__golden__`` entry backed by the same compute core."""
    del args
    return _kernel_golden(
        x,
        gamma,
        beta,
        mean,
        variance,
        epsilon=epsilon,
        **kwargs,
    )


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt declares ``ACLNNTYPE aclnn_exclude``;
# the repository provides neither op_api files nor an aclnn interface document.
# 【不存在】e2e 通路：the supported surface is GE graph mode (README.md), and no
# torch binding for this operator is provided in the repository.
