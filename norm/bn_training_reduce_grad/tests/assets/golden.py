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

"""BNTrainingReduceGrad kernel/GEIR golden in the TestSpec multi-path format.

For an ND input laid out as ``[N, C, R...]`` with ``num = N * R`` the operator
computes per channel ``c``::

    sqrt_var[c]    = sqrt(batch_variance[c] + epsilon)
    multiplier[c]  = (diff_scale[c] * (-1/num)) / sqrt_var[c]
    addend[c]      = (batch_mean[c] / sqrt_var[c]) * (diff_scale[c] * (1/num))
                     + diff_offset[c] * (-1/num)
    mul_scale[c]   = scale[c] / sqrt_var[c]
    y[n, c, r]     = ((grads[n, c, r] + multiplier[c] * x[n, c, r]) + addend[c]) * mul_scale[c]

The CPU true-value path is a Torch competitor composition.  It lifts fp16/bf16
to at least fp32 and preserves fp64 inputs supplied by TTK Promote; promoted
values are never narrowed.  ``numRecip`` is quantized to fp32 before use (and
``negNumRecip`` is its fp32 negation), the same rounding the host tiling applies
(fp64 computed then cast to fp32, matching the A2 TBE ``tvm.const`` semantics).
The independent third-party composition mirrors the arch35 kernel's float32
arithmetic and operation order (sqrt followed by IEEE division, coefficient
combination order mul -> add(grads) -> add(addend) -> mul(mul_scale)) before
casting ``y`` back to the input dtype.
"""

import numpy as np
import torch


# Kernel and GEIR resolve the same snake-case operator key and share one Spec.
__spec__ = {"bn_training_reduce_grad": "BNTrainingReduceGradKernelSpec"}

# Compatibility entry for the historical kernel golden loader.
__golden__ = {
    "kernel": {"bn_training_reduce_grad": "bn_training_reduce_grad_golden"},
}


_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
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
    return float(_attr(values, "epsilon", 0.0001))


def _as_tensor(value):
    """Convert a Kernel/GEIR NumPy input to a CPU Torch tensor losslessly."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    arr = np.ascontiguousarray(np.asarray(value))
    # torch.from_numpy does not accept ml_dtypes.bfloat16; fp32 round-trip is lossless.
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def _reference_dtype(*tensors):
    """Select at least fp32 while retaining any wider promoted float dtype."""
    dtype = torch.float32
    for tensor in tensors:
        if tensor is not None and tensor.dtype.is_floating_point:
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _num_recip_f32(n, r):
    """fp64 reciprocal of N*R rounded to fp32, mirroring the host tiling."""
    return float(
        torch.tensor(1.0 / float(n * r), dtype=torch.float64).to(torch.float32).item()
    )


def _stat_vector(tensor, c, name):
    if tensor.numel() != c:
        raise ValueError(f"{name} must contain C={c} elements, got {tensor.numel()}")
    return torch.reshape(tensor, (c,))


def _compute(
    grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, epsilon
):
    """Sole Torch true-value core; returns ``[y]`` in def.cpp order."""
    grads_tensor = _as_tensor(grads)
    x_tensor = _as_tensor(x)
    diff_scale_tensor = _as_tensor(diff_scale)
    diff_offset_tensor = _as_tensor(diff_offset)
    scale_tensor = _as_tensor(scale)
    batch_mean_tensor = _as_tensor(batch_mean)
    batch_variance_tensor = _as_tensor(batch_variance)

    if grads_tensor.ndim < 2:
        raise ValueError(f"grads rank must be at least 2, got {grads_tensor.ndim}")
    n, c = grads_tensor.shape[:2]
    r = 1
    for dim in grads_tensor.shape[2:]:
        r *= dim

    compute_dtype = _reference_dtype(
        grads_tensor,
        x_tensor,
        diff_scale_tensor,
        diff_offset_tensor,
        scale_tensor,
        batch_mean_tensor,
        batch_variance_tensor,
    )
    grads_compute = grads_tensor.to(dtype=compute_dtype)
    x_compute = x_tensor.to(dtype=compute_dtype)
    diff_scale_compute = _stat_vector(
        diff_scale_tensor.to(dtype=compute_dtype), c, "diff_scale"
    )
    diff_offset_compute = _stat_vector(
        diff_offset_tensor.to(dtype=compute_dtype), c, "diff_offset"
    )
    scale_compute = _stat_vector(scale_tensor.to(dtype=compute_dtype), c, "scale")
    batch_mean_compute = _stat_vector(
        batch_mean_tensor.to(dtype=compute_dtype), c, "batch_mean"
    )
    batch_variance_compute = _stat_vector(
        batch_variance_tensor.to(dtype=compute_dtype), c, "batch_variance"
    )

    # The host stores numRecip/negNumRecip/epsilon as fp32 before kernel launch;
    # keep that fp32 quantization even when Promote lifts to fp64.
    num_recip = torch.tensor(_num_recip_f32(n, r), dtype=torch.float32).to(
        compute_dtype
    )
    neg_num_recip = torch.neg(num_recip)
    epsilon_tensor = torch.tensor(float(epsilon), dtype=torch.float32).to(compute_dtype)

    sqrt_var = torch.sqrt(torch.add(batch_variance_compute, epsilon_tensor))
    multiplier = torch.div(torch.mul(diff_scale_compute, neg_num_recip), sqrt_var)
    addend = torch.add(
        torch.mul(
            torch.div(batch_mean_compute, sqrt_var),
            torch.mul(diff_scale_compute, num_recip),
        ),
        torch.mul(diff_offset_compute, neg_num_recip),
    )
    mul_scale = torch.div(scale_compute, sqrt_var)

    broadcast_shape = (1, c) + (1,) * (grads_tensor.ndim - 2)
    y = torch.mul(
        torch.add(
            torch.add(
                grads_compute,
                torch.mul(torch.reshape(multiplier, broadcast_shape), x_compute),
            ),
            torch.reshape(addend, broadcast_shape),
        ),
        torch.reshape(mul_scale, broadcast_shape),
    )

    return [y]


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
    grads,
    x,
    diff_scale,
    diff_offset,
    scale,
    batch_mean,
    batch_variance,
    epsilon=0.0001,
    **kwargs,
):
    """Kernel/GEIR adapter: NumPy inputs and a NumPy output list."""
    epsilon_value = _resolve_epsilon(epsilon, kwargs)
    outputs = _compute(
        grads,
        x,
        diff_scale,
        diff_offset,
        scale,
        batch_mean,
        batch_variance,
        epsilon_value,
    )
    output_dtypes = kwargs.get("output_dtypes")
    if not output_dtypes:
        output_dtypes = (_input_dtype_name(grads),)
    return _numpy_outputs(outputs, output_dtypes)


class _BNTrainingReduceGradCompose:
    """Independent Torch composition matching the arch35 device arithmetic.

    性能腿按竞品最优形态执行:torch.compile(dynamic=True) 融合,编译失败自动回落 eager
    (三方性能倍数不虚高的关键,实测可差 5 倍以上)。
    """

    def __init__(self, epsilon=0.0001, **kwargs):
        epsilon_value = _resolve_epsilon(epsilon, kwargs)
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(epsilon_value, dtype=torch.float32).item())
        self._compiled = None

    def _impl(
        self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    ):
        if grads.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                f"BNTrainingReduceGrad supports only float16/float32/bfloat16 grads, got {grads.dtype}"
            )

        n, c = grads.shape[:2]
        r = 1
        for dim in grads.shape[2:]:
            r *= dim
        broadcast_shape = (1, c) + (1,) * (grads.ndim - 2)

        # BNTrainingReduceGradKernel::Process converts every arithmetic operand to fp32.
        grads_f32 = grads.to(dtype=torch.float32)
        x_f32 = x.to(dtype=torch.float32)
        diff_scale_f32 = _stat_vector(
            diff_scale.to(dtype=torch.float32), c, "diff_scale"
        )
        diff_offset_f32 = _stat_vector(
            diff_offset.to(dtype=torch.float32), c, "diff_offset"
        )
        scale_f32 = _stat_vector(scale.to(dtype=torch.float32), c, "scale")
        batch_mean_f32 = _stat_vector(
            batch_mean.to(dtype=torch.float32), c, "batch_mean"
        )
        batch_variance_f32 = _stat_vector(
            batch_variance.to(dtype=torch.float32), c, "batch_variance"
        )
        num_recip = torch.tensor(
            _num_recip_f32(n, r), dtype=torch.float32, device=grads.device
        )
        neg_num_recip = torch.neg(num_recip)
        epsilon_f32 = torch.tensor(
            self.epsilon, dtype=torch.float32, device=grads.device
        )

        sqrt_var = torch.sqrt(torch.add(batch_variance_f32, epsilon_f32))
        multiplier = torch.div(torch.mul(diff_scale_f32, neg_num_recip), sqrt_var)
        addend = torch.add(
            torch.mul(
                torch.div(batch_mean_f32, sqrt_var),
                torch.mul(diff_scale_f32, num_recip),
            ),
            torch.mul(diff_offset_f32, neg_num_recip),
        )
        mul_scale = torch.div(scale_f32, sqrt_var)
        y_f32 = torch.mul(
            torch.add(
                torch.add(
                    grads_f32,
                    torch.mul(torch.reshape(multiplier, broadcast_shape), x_f32),
                ),
                torch.reshape(addend, broadcast_shape),
            ),
            torch.reshape(mul_scale, broadcast_shape),
        )

        # The sole output is stored in grads dtype.
        return [y_f32.to(dtype=grads.dtype)]

    def __call__(
        self,
        grads,
        x,
        diff_scale,
        diff_offset,
        scale,
        batch_mean,
        batch_variance,
        **kwargs,
    ):
        del kwargs
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(
                grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
            )
        except Exception:
            self._compiled = self._impl
            return self._impl(
                grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
            )


class BNTrainingReduceGradKernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow bn_training_reduce_grad_def.cpp."""

    golden = _kernel_golden
    third_party = {"torch": _BNTrainingReduceGradCompose}
    tolerance = _TOL


def bn_training_reduce_grad_golden(
    grads,
    x,
    diff_scale,
    diff_offset,
    scale,
    batch_mean,
    batch_variance,
    epsilon=0.0001,
    *args,
    **kwargs,
):
    """Compatibility ``__golden__`` entry backed by the same compute core."""
    del args
    return _kernel_golden(
        grads,
        x,
        diff_scale,
        diff_offset,
        scale,
        batch_mean,
        batch_variance,
        epsilon=epsilon,
        **kwargs,
    )


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt declares ``ACLNNTYPE aclnn_exclude``;
# the repository provides neither op_api files nor an aclnn interface document.
# 【不存在】e2e 通路：canndev 无 BNTrainingReduceGrad 的 aclnn 接口
# （该算子为 GE 图内 FusedBatchNormGrad 融合展开产物），torch_npu 无单算子调用。
# 【不存在】tf/onnx 端到端通路：canndev framework/ 有 tf_plugin（A2 TF 图映射），
# 但 A5 交付不含 framework 插件源。
