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

"""BNTrainingUpdateGrad kernel/GEIR golden in the TestSpec multi-path format.

For ND inputs laid out as ``[N, C, R...]`` the operator computes per channel
``c`` (reduction over the N axis and all trailing R axes)::

    rstd[c]         = 1 / sqrt(batch_variance[c] + epsilon)
    diff_scale[c]   = sum_{n,r} grads[n,c,r] * (x[n,c,r] - batch_mean[c]) * rstd[c]
    diff_offset[c]  = sum_{n,r} grads[n,c,r]

The CPU true-value path is a Torch competitor composition.  It lifts fp16/bf16
to at least fp32 and preserves fp64 inputs supplied by TTK Promote; promoted
values are never narrowed.  The independent third-party composition mirrors the
arch35 kernel's float32 arithmetic and operation order (sub -> mul rstd ->
mul grads, matching the A2 TBE ``(x - mean) * rstd`` then ``grads * x_norm``
chain); both outputs are always float32.
"""

import numpy as np
import torch


# Kernel and GEIR resolve the same snake-case operator key and share one Spec.
__spec__ = {"bn_training_update_grad": "BNTrainingUpdateGradKernelSpec"}

# Compatibility entry for the historical kernel golden loader.
__golden__ = {
    "kernel": {"bn_training_update_grad": "bn_training_update_grad_golden"},
}


_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_DEFAULT_EPSILON = 0.0001  # A2 proto .ATTR(epsilon, Float, 0.0001)


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
    return float(_attr(values, "epsilon", _DEFAULT_EPSILON))


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


def _stat_vector(tensor, c, name):
    if tensor.numel() != c:
        raise ValueError(f"{name} must contain C={c} elements, got {tensor.numel()}")
    return torch.reshape(tensor, (c,))


def _compute(grads, x, batch_mean, batch_variance, epsilon):
    """Sole Torch true-value core; return outputs in def.cpp order."""
    grads_tensor = _as_tensor(grads)
    x_tensor = _as_tensor(x)
    mean_tensor = _as_tensor(batch_mean)
    var_tensor = _as_tensor(batch_variance)

    if grads_tensor.ndim < 2:
        raise ValueError(f"grads rank must be at least 2, got {grads_tensor.ndim}")
    c = grads_tensor.shape[1]

    compute_dtype = _reference_dtype(grads_tensor, x_tensor, mean_tensor, var_tensor)
    grads_compute = grads_tensor.to(dtype=compute_dtype)
    x_compute = x_tensor.to(dtype=compute_dtype)
    mean_compute = _stat_vector(mean_tensor.to(dtype=compute_dtype), c, "batch_mean")
    var_compute = _stat_vector(var_tensor.to(dtype=compute_dtype), c, "batch_variance")

    # Tiling stores epsilon as float32 before kernel launch; keep that fp32
    # quantization even when Promote lifts to fp64.
    epsilon_tensor = torch.tensor(float(epsilon), dtype=torch.float32).to(compute_dtype)

    rstd = torch.div(
        torch.ones_like(var_compute), torch.sqrt(torch.add(var_compute, epsilon_tensor))
    )
    broadcast_shape = (1, c) + (1,) * (grads_tensor.ndim - 2)
    mean_bcast = torch.reshape(mean_compute, broadcast_shape)
    rstd_bcast = torch.reshape(rstd, broadcast_shape)

    x_norm = torch.mul(torch.sub(x_compute, mean_bcast), rstd_bcast)
    scale_mul = torch.mul(grads_compute, x_norm)

    reduce_dims = (0,) + tuple(range(2, grads_tensor.ndim))
    diff_scale = torch.sum(scale_mul, dim=reduce_dims)
    diff_offset = torch.sum(grads_compute, dim=reduce_dims)
    return [diff_scale, diff_offset]


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
        "bf16": "bfloat16",
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
    grads, x, batch_mean, batch_variance, epsilon=_DEFAULT_EPSILON, **kwargs
):
    """Kernel/GEIR adapter: NumPy inputs and a NumPy output list."""
    epsilon_value = _resolve_epsilon(epsilon, kwargs)
    outputs = _compute(grads, x, batch_mean, batch_variance, epsilon_value)
    output_dtypes = kwargs.get("output_dtypes")
    if not output_dtypes:
        output_dtypes = ("float32", "float32")  # 两路输出恒 fp32
    return _numpy_outputs(outputs, output_dtypes)


class _BNTrainingUpdateGradCompose:
    """Independent Torch composition matching the arch35 device arithmetic.

    性能腿按竞品最优形态执行:torch.compile(dynamic=True) 融合,编译失败自动回落 eager
    (三方性能倍数不虚高的关键,实测可差 5 倍以上)。
    """

    def __init__(self, epsilon=_DEFAULT_EPSILON, **kwargs):
        epsilon_value = _resolve_epsilon(epsilon, kwargs)
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(epsilon_value, dtype=torch.float32).item())
        self._compiled = None

    def _impl(self, grads, x, batch_mean, batch_variance):
        if grads.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                f"BNTrainingUpdateGrad supports only float16/float32/bfloat16 grads, got {grads.dtype}"
            )

        c = grads.shape[1]
        broadcast_shape = (1, c) + (1,) * (grads.ndim - 2)

        # BNTrainingUpdateGradKernel converts every arithmetic operand to fp32.
        grads_f32 = grads.to(dtype=torch.float32)
        x_f32 = x.to(dtype=torch.float32)
        mean_f32 = _stat_vector(batch_mean.to(dtype=torch.float32), c, "batch_mean")
        var_f32 = _stat_vector(
            batch_variance.to(dtype=torch.float32), c, "batch_variance"
        )
        epsilon_f32 = torch.tensor(
            self.epsilon, dtype=torch.float32, device=grads.device
        )

        rstd = torch.div(
            torch.ones_like(var_f32), torch.sqrt(torch.add(var_f32, epsilon_f32))
        )
        x_norm = torch.mul(
            torch.sub(x_f32, torch.reshape(mean_f32, broadcast_shape)),
            torch.reshape(rstd, broadcast_shape),
        )
        scale_mul = torch.mul(grads_f32, x_norm)

        reduce_dims = (0,) + tuple(range(2, grads.ndim))
        diff_scale = torch.sum(scale_mul, dim=reduce_dims)
        diff_offset = torch.sum(grads_f32, dim=reduce_dims)
        return [diff_scale, diff_offset]  # 两路输出恒 fp32

    def __call__(self, grads, x, batch_mean, batch_variance, **kwargs):
        del kwargs
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(grads, x, batch_mean, batch_variance)
        except Exception:
            self._compiled = self._impl
            return self._impl(grads, x, batch_mean, batch_variance)


class BNTrainingUpdateGradKernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow bn_training_update_grad_def.cpp."""

    golden = _kernel_golden
    third_party = {"torch": _BNTrainingUpdateGradCompose}
    tolerance = _TOL


def bn_training_update_grad_golden(
    grads, x, batch_mean, batch_variance, epsilon=_DEFAULT_EPSILON, *args, **kwargs
):
    """Compatibility ``__golden__`` entry backed by the same compute core."""
    del args
    return _kernel_golden(
        grads,
        x,
        batch_mean,
        batch_variance,
        epsilon=epsilon,
        **kwargs,
    )


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt declares ``ACLNNTYPE aclnn_exclude``;
# the repository provides neither op_api files nor an aclnn interface document.
# 【不存在】e2e 通路：strings <torch_npu 全部 .so> | grep -c aclnnBNTrainingUpdateGrad = 0
# (本机 torch_npu 2.7.1 实测), torch_npu never invokes this operator's aclnn.
# 【不涉及】tf 端到端通路：canndev 有 BNTrainingUpdateGrad tf 插件,源已逐字镜像保留在
# framework/(对齐 A2 组织与同族 V2 做法),tf 端到端验证本次产品规格不涉及。
# 【不存在】onnx 通路：canndev framework/onnx_plugin/ grep 无 BNTrainingUpdateGrad。
