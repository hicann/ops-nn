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

"""BNTrainingUpdateV2 kernel/GEIR golden in the TestSpec multi-path format.

For an ND input laid out as ``[N, C, R...]`` with ``num = N * R`` the operator
computes per channel ``c``::

    batch_mean[c]     = sum[c] / num
    batch_variance[c] = square_sum[c] / num - batch_mean[c] ** 2
    multiplier[c]     = scale[c] / sqrt(batch_variance[c] + epsilon)
    addend[c]         = offset[c] - multiplier[c] * batch_mean[c]
    y[n, c, r]        = multiplier[c] * x[n, c, r] + addend[c]

The CPU true-value path is a Torch competitor composition.  It lifts fp16/bf16
to at least fp32 and preserves fp64 inputs supplied by TTK Promote; promoted
values are never narrowed.  ``numRecip`` is quantized to fp32 before use, the
same rounding the host tiling applies (fp64 reciprocal cast to fp32, matching
the A2 TBE ``tvm.const`` semantics).  The independent third-party composition
mirrors the arch35 kernel's float32 arithmetic and operation order before
casting ``y`` back to the input dtype.
"""

import numpy as np
import torch


# Kernel and GEIR resolve the same snake-case operator key and share one Spec.
__spec__ = {"bn_training_update_v2": "BNTrainingUpdateV2KernelSpec"}

# Compatibility entry for the historical kernel golden loader.
__golden__ = {
    "kernel": {"bn_training_update_v2": "bn_training_update_v2_golden"},
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


def _num_recip_f32(n, r):
    """fp64 reciprocal of N*R rounded to fp32, mirroring the host tiling."""
    return float(
        torch.tensor(1.0 / float(n * r), dtype=torch.float64).to(torch.float32).item()
    )


def _stat_vector(tensor, c, name):
    if tensor.numel() != c:
        raise ValueError(f"{name} must contain C={c} elements, got {tensor.numel()}")
    return torch.reshape(tensor, (c,))


def _compute(x, sum, square_sum, scale, offset, epsilon):
    """Sole Torch true-value core; return outputs in def.cpp order."""
    x_tensor = _as_tensor(x)
    sum_tensor = _as_tensor(sum)
    square_sum_tensor = _as_tensor(square_sum)
    scale_tensor = _as_tensor(scale)
    offset_tensor = _as_tensor(offset)

    if x_tensor.ndim < 2:
        raise ValueError(f"x rank must be at least 2, got {x_tensor.ndim}")
    n, c = x_tensor.shape[:2]
    r = 1
    for dim in x_tensor.shape[2:]:
        r *= dim

    compute_dtype = _reference_dtype(
        x_tensor, sum_tensor, square_sum_tensor, scale_tensor, offset_tensor
    )
    x_compute = x_tensor.to(dtype=compute_dtype)
    sum_compute = _stat_vector(sum_tensor.to(dtype=compute_dtype), c, "sum")
    square_sum_compute = _stat_vector(
        square_sum_tensor.to(dtype=compute_dtype), c, "square_sum"
    )
    scale_compute = _stat_vector(scale_tensor.to(dtype=compute_dtype), c, "scale")
    offset_compute = _stat_vector(offset_tensor.to(dtype=compute_dtype), c, "offset")

    # The host stores numRecip and epsilon as fp32 before kernel launch; keep
    # that fp32 quantization even when Promote has lifted tensors to fp64.
    num_recip = torch.tensor(_num_recip_f32(n, r), dtype=torch.float32).to(
        compute_dtype
    )
    epsilon_tensor = torch.tensor(float(epsilon), dtype=torch.float32).to(compute_dtype)

    batch_mean = torch.mul(sum_compute, num_recip)
    batch_variance = torch.sub(
        torch.mul(square_sum_compute, num_recip), torch.mul(batch_mean, batch_mean)
    )
    multiplier = torch.div(
        scale_compute, torch.sqrt(torch.add(batch_variance, epsilon_tensor))
    )
    addend = torch.sub(offset_compute, torch.mul(multiplier, batch_mean))

    broadcast_shape = (1, c) + (1,) * (x_tensor.ndim - 2)
    y = torch.add(
        torch.mul(x_compute, torch.reshape(multiplier, broadcast_shape)),
        torch.reshape(addend, broadcast_shape),
    )

    return [y, torch.clone(batch_mean), torch.clone(batch_variance)]


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


def _kernel_golden(x, sum, square_sum, scale, offset, epsilon=1e-5, **kwargs):
    """Kernel/GEIR adapter: NumPy inputs and a NumPy output list."""
    epsilon_value = _resolve_epsilon(epsilon, kwargs)
    outputs = _compute(x, sum, square_sum, scale, offset, epsilon_value)
    output_dtypes = kwargs.get("output_dtypes")
    if not output_dtypes:
        output_dtypes = (
            _input_dtype_name(x),
            _input_dtype_name(scale),
            _input_dtype_name(scale),
        )
    return _numpy_outputs(outputs, output_dtypes)


class _BNTrainingUpdateV2Compose:
    """Independent Torch composition matching the arch35 device arithmetic.

    性能腿按竞品最优形态执行:torch.compile(dynamic=True) 融合,编译失败自动回落 eager
    (三方性能倍数不虚高的关键,实测可差 5 倍以上)。
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        epsilon_value = _resolve_epsilon(epsilon, kwargs)
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(epsilon_value, dtype=torch.float32).item())
        self._compiled = None

    def _impl(self, x, sum, square_sum, scale, offset):
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                f"BNTrainingUpdateV2 supports only float16/float32/bfloat16 x, got {x.dtype}"
            )

        n, c = x.shape[:2]
        r = 1
        for dim in x.shape[2:]:
            r *= dim
        broadcast_shape = (1, c) + (1,) * (x.ndim - 2)

        # BNTrainingUpdateV2Kernel::Process converts every arithmetic operand to fp32.
        x_f32 = x.to(dtype=torch.float32)
        sum_f32 = _stat_vector(sum.to(dtype=torch.float32), c, "sum")
        square_sum_f32 = _stat_vector(
            square_sum.to(dtype=torch.float32), c, "square_sum"
        )
        scale_f32 = _stat_vector(scale.to(dtype=torch.float32), c, "scale")
        offset_f32 = _stat_vector(offset.to(dtype=torch.float32), c, "offset")
        num_recip = torch.tensor(
            _num_recip_f32(n, r), dtype=torch.float32, device=x.device
        )
        epsilon_f32 = torch.tensor(self.epsilon, dtype=torch.float32, device=x.device)

        batch_mean = torch.mul(sum_f32, num_recip)
        batch_variance = torch.sub(
            torch.mul(square_sum_f32, num_recip), torch.mul(batch_mean, batch_mean)
        )
        multiplier = torch.div(
            scale_f32, torch.sqrt(torch.add(batch_variance, epsilon_f32))
        )
        addend = torch.sub(offset_f32, torch.mul(multiplier, batch_mean))
        y_f32 = torch.add(
            torch.mul(x_f32, torch.reshape(multiplier, broadcast_shape)),
            torch.reshape(addend, broadcast_shape),
        )

        # The first output is stored in x dtype; the two stat outputs are fp32.
        return [
            y_f32.to(dtype=x.dtype),
            batch_mean.clone(),
            batch_variance.clone(),
        ]

    def __call__(self, x, sum, square_sum, scale, offset, **kwargs):
        del kwargs
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(x, sum, square_sum, scale, offset)
        except Exception:
            self._compiled = self._impl
            return self._impl(x, sum, square_sum, scale, offset)


class BNTrainingUpdateV2KernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow bn_training_update_v2_def.cpp."""

    golden = _kernel_golden
    third_party = {"torch": _BNTrainingUpdateV2Compose}
    tolerance = _TOL


def bn_training_update_v2_golden(
    x, sum, square_sum, scale, offset, epsilon=1e-5, *args, **kwargs
):
    """Compatibility ``__golden__`` entry backed by the same compute core."""
    del args
    return _kernel_golden(
        x,
        sum,
        square_sum,
        scale,
        offset,
        epsilon=epsilon,
        **kwargs,
    )


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt declares ``ACLNNTYPE aclnn_exclude``;
# the repository provides neither op_api files nor an aclnn interface document.
# 【不存在】e2e 通路：strings libtorch_npu.so | grep -c aclnnBNTrainingUpdateV2 = 0
# (torch_npu 2.7.1 实测), torch_npu never invokes this operator's aclnn.
# 【不存在】tf/onnx 端到端通路：framework/ 仅逐字 mirror canndev 的 tf plugin 注册源,
# tf 端到端验证本次产品规格不涉及;canndev 无 onnx 插件。
