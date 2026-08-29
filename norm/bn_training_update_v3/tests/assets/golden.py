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

"""BNTrainingUpdateV3 kernel/GEIR golden in the TestSpec multi-path format.

For an ND input laid out as ``[N, C, R...]`` (or an NHWC input of any rank >= 2
with ``C`` as the last dim) and ``num = numel / C`` the operator computes per
channel ``c``::

    save_mean[c]      = sum[c] / num
    save_variance[c]  = square_sum[c] / num - save_mean[c] ** 2
    batch_mean[c]     = save_mean[c]
    batch_variance[c] = save_variance[c] * num / (num - 1)   (0.0 when num == 1)
    reserve_1[c]      = save_mean[c]
    reserve_2[c]      = save_variance[c]
    multiplier[c]     = scale[c] / sqrt(save_variance[c] + epsilon)
    addend[c]         = offset[c] - multiplier[c] * save_mean[c]
    y[..., c, ...]    = multiplier[c] * x[..., c, ...] + addend[c]

The x format is taken from TTK's ``input_formats`` kwarg when present (the
authoritative origin-format declaration); otherwise a heuristic keeps ND
unless the last dim matches the statistics length and dim1 does not.

The CPU true-value path is a Torch competitor composition.  It lifts fp16/bf16
to at least fp32 and preserves fp64 inputs supplied by TTK Promote; promoted
values are never narrowed.  ``numRecip`` and ``batchVarScaler`` are quantized
to fp32 before use, the same rounding the host tiling applies (fp64 computed
then cast to fp32, matching the A2 TBE ``tvm.const`` / python-float semantics).
The independent third-party composition mirrors the arch35 kernel's float32
arithmetic and operation order before casting ``y`` back to the input dtype.
"""

import numpy as np
import torch


# Kernel and GEIR resolve the same snake-case operator key and share one Spec.
__spec__ = {"bn_training_update_v3": "BNTrainingUpdateV3KernelSpec"}

# Compatibility entry for the historical kernel golden loader.
__golden__ = {
    "kernel": {"bn_training_update_v3": "bn_training_update_v3_golden"},
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
    return _num_recip_f32_num(n * r)


def _num_recip_f32_num(num):
    """fp64 reciprocal of num rounded to fp32 (ND num=N*R, NHWC num=numel/C)."""
    return float(
        torch.tensor(1.0 / float(num), dtype=torch.float64).to(torch.float32).item()
    )


def _batch_var_scaler_f32(n, r):
    """fp64 num/(num-1) rounded to fp32 (0.0 when num == 1), mirroring the host."""
    return _batch_var_scaler_f32_num(n * r)


def _batch_var_scaler_f32_num(num):
    """fp64 num/(num-1) rounded to fp32 (0.0 when num == 1)."""
    if num == 1:
        return 0.0
    return float(
        torch.tensor(float(num) / float(num - 1), dtype=torch.float64)
        .to(torch.float32)
        .item()
    )


def _infer_x_format(x, sum, input_formats):
    """ND by default; prefer TTK's authoritative input_formats declaration."""
    if input_formats:
        return (
            input_formats[0]
            if isinstance(input_formats, (list, tuple))
            else input_formats
        )
    # Fallback heuristic (remote third-party provider sends tensors only):
    # last dim matches the statistics length and dim1 does not -> NHWC.  Rank-2
    # [rows, C] is layout-identical under ND and NHWC, so the ambiguity is moot.
    stat_len = (
        int(np.asarray(sum).size) if not isinstance(sum, torch.Tensor) else sum.numel()
    )
    if stat_len > 0:
        shape = tuple(x.shape)
        if shape[-1] == stat_len and shape[1] != stat_len:
            return "NHWC"
    return "ND"


def _stat_vector(tensor, c, name):
    if tensor.numel() != c:
        raise ValueError(f"{name} must contain C={c} elements, got {tensor.numel()}")
    return torch.reshape(tensor, (c,))


def _compute(x, sum, square_sum, scale, offset, epsilon, x_format="ND"):
    """Sole Torch true-value core; return outputs in def.cpp order."""
    x_tensor = _as_tensor(x)
    sum_tensor = _as_tensor(sum)
    square_sum_tensor = _as_tensor(square_sum)
    scale_tensor = _as_tensor(scale)
    offset_tensor = _as_tensor(offset)

    if x_tensor.ndim < 2:
        raise ValueError(f"x rank must be at least 2, got {x_tensor.ndim}")
    is_nhwc = str(x_format).upper() == "NHWC"
    if is_nhwc:
        c = x_tensor.shape[-1]
        broadcast_shape = (1,) * (x_tensor.ndim - 1) + (c,)
    else:
        c = x_tensor.shape[1]
        broadcast_shape = (1, c) + (1,) * (x_tensor.ndim - 2)
    num = x_tensor.numel() // c  # ND: N*R；NHWC: numel/C，数值一致

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

    # The host stores numRecip/batchVarScaler/epsilon as fp32 before kernel
    # launch; keep that fp32 quantization even when Promote lifts to fp64.
    num_recip = torch.tensor(_num_recip_f32_num(num), dtype=torch.float32).to(
        compute_dtype
    )
    batch_var_scaler = torch.tensor(
        _batch_var_scaler_f32_num(num), dtype=torch.float32
    ).to(compute_dtype)
    epsilon_tensor = torch.tensor(float(epsilon), dtype=torch.float32).to(compute_dtype)

    save_mean = torch.mul(sum_compute, num_recip)
    save_variance = torch.sub(
        torch.mul(square_sum_compute, num_recip), torch.mul(save_mean, save_mean)
    )
    batch_variance = torch.mul(save_variance, batch_var_scaler)
    multiplier = torch.div(
        scale_compute, torch.sqrt(torch.add(save_variance, epsilon_tensor))
    )
    addend = torch.sub(offset_compute, torch.mul(multiplier, save_mean))

    y = torch.add(
        torch.mul(x_compute, torch.reshape(multiplier, broadcast_shape)),
        torch.reshape(addend, broadcast_shape),
    )

    return [
        y,
        torch.clone(save_mean),  # batch_mean
        torch.clone(batch_variance),  # batch_variance (unbiased)
        torch.clone(save_mean),  # reserve_1
        torch.clone(save_variance),  # reserve_2
    ]


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
    x_format = _infer_x_format(x, sum, kwargs.get("input_formats"))
    outputs = _compute(x, sum, square_sum, scale, offset, epsilon_value, x_format)
    output_dtypes = kwargs.get("output_dtypes")
    if not output_dtypes:
        stat_dtype = _input_dtype_name(scale)
        output_dtypes = (
            _input_dtype_name(x),
            stat_dtype,
            stat_dtype,
            stat_dtype,
            stat_dtype,
        )
    return _numpy_outputs(outputs, output_dtypes)


class _BNTrainingUpdateV3Compose:
    """Independent Torch composition matching the arch35 device arithmetic.

    性能腿按竞品最优形态执行:torch.compile(dynamic=True) 融合,编译失败自动回落 eager
    (三方性能倍数不虚高的关键,实测可差 5 倍以上)。
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        epsilon_value = _resolve_epsilon(epsilon, kwargs)
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(epsilon_value, dtype=torch.float32).item())
        self._compiled = None

    def _impl(self, x, sum, square_sum, scale, offset, x_format="ND"):
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                f"BNTrainingUpdateV3 supports only float16/float32/bfloat16 x, got {x.dtype}"
            )

        is_nhwc = str(x_format).upper() == "NHWC"
        if is_nhwc:
            c = x.shape[-1]
            broadcast_shape = (1,) * (x.ndim - 1) + (c,)
        else:
            c = x.shape[1]
            broadcast_shape = (1, c) + (1,) * (x.ndim - 2)
        num = x.numel() // c  # ND: N*R；NHWC: numel/C

        # BNTrainingUpdateV3Kernel::Process converts every arithmetic operand to fp32.
        x_f32 = x.to(dtype=torch.float32)
        sum_f32 = _stat_vector(sum.to(dtype=torch.float32), c, "sum")
        square_sum_f32 = _stat_vector(
            square_sum.to(dtype=torch.float32), c, "square_sum"
        )
        scale_f32 = _stat_vector(scale.to(dtype=torch.float32), c, "scale")
        offset_f32 = _stat_vector(offset.to(dtype=torch.float32), c, "offset")
        num_recip = torch.tensor(
            _num_recip_f32_num(num), dtype=torch.float32, device=x.device
        )
        batch_var_scaler = torch.tensor(
            _batch_var_scaler_f32_num(num), dtype=torch.float32, device=x.device
        )
        epsilon_f32 = torch.tensor(self.epsilon, dtype=torch.float32, device=x.device)

        save_mean = torch.mul(sum_f32, num_recip)
        save_variance = torch.sub(
            torch.mul(square_sum_f32, num_recip), torch.mul(save_mean, save_mean)
        )
        batch_variance = torch.mul(save_variance, batch_var_scaler)
        multiplier = torch.div(
            scale_f32, torch.sqrt(torch.add(save_variance, epsilon_f32))
        )
        addend = torch.sub(offset_f32, torch.mul(multiplier, save_mean))
        y_f32 = torch.add(
            torch.mul(x_f32, torch.reshape(multiplier, broadcast_shape)),
            torch.reshape(addend, broadcast_shape),
        )

        # The first output is stored in x dtype; the four stat outputs are fp32.
        return [
            y_f32.to(dtype=x.dtype),
            save_mean.clone(),
            batch_variance.clone(),
            save_mean.clone(),
            save_variance.clone(),
        ]

    def __call__(self, x, sum, square_sum, scale, offset, **kwargs):
        # 在丢弃 kwargs 前抓取 TTK 注入的 x 格式声明（缺省 ND）
        x_format = _infer_x_format(x, sum, kwargs.get("input_formats"))
        del kwargs
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(x, sum, square_sum, scale, offset, x_format)
        except Exception:
            self._compiled = self._impl
            return self._impl(x, sum, square_sum, scale, offset, x_format)


class BNTrainingUpdateV3KernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow bn_training_update_v3_def.cpp."""

    golden = _kernel_golden
    third_party = {"torch": _BNTrainingUpdateV3Compose}
    tolerance = _TOL


def bn_training_update_v3_golden(
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
# 【不存在】e2e 通路：strings libtorch_npu.so | grep -c aclnnBNTrainingUpdateV3 = 0
# (torch_npu 2.7.1 实测), torch_npu never invokes this operator's aclnn.
# 【不存在】tf/onnx 端到端通路：canndev framework/ 无 BNTrainingUpdateV3 的
# tf/onnx 插件(grep 实测),故本算子不镜像 framework/ 插件源。
