# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend 950 AddRmsNormQuant references, independent of the TTK package.

Kernel and GEIR use the native schema; ACLNN uses the public API schema.
The NumPy golden retains the FP32 residual sum until the final xOut cast.
The independent PyTorch reference computes normalization in FP64 on CPU.
"""

# TTK: pass this file as --plugin and use --pc=1. The public torch_npu E2E
# adapter for both versions is in add_rms_norm_quant_v2/tests/assets/golden.py.

import numpy as np
import torch
import ml_dtypes
from en_dtypes import hifloat8

__spec__ = {
    "add_rms_norm_quant": "AddRmsNormQuantTestSpec",
    "aclnnAddRmsNormQuant": "AclnnAddRmsNormQuantTestSpec",
}

_DST_TYPES = {2: "int8", 34: "hifloat8", 35: "float8_e5m2", 36: "float8_e4m3fn"}
_NUMPY_TYPES = {
    "int8": np.int8,
    "hifloat8": hifloat8,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
}
_SATURATION = {
    "int8": (-128, 127),
    "hifloat8": (-32768, 32768),
    "float8_e5m2": (-57344, 57344),
    "float8_e4m3fn": (-448, 448),
}
# Mixed output dtypes require per-dtype tolerances. TTK must support TestSpec
# requant and semantic HIFLOAT8 comparison; do not replace this map with a
# global requant comparison.
_TOLERANCE = {
    "int8": {"standard": "quant"},
    "hifloat8": {"standard": "requant"},
    "float8_e5m2": {"standard": "requant"},
    "float8_e4m3fn": {"standard": "requant"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
}


def _numpy(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        if value.dtype == torch.float8_e5m2:
            return value.view(torch.uint8).numpy().view(ml_dtypes.float8_e5m2)
        if value.dtype == torch.float8_e4m3fn:
            return value.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
        return value.numpy()
    return np.asarray(value)


def _torch(value):
    if value is None:
        return None
    value = np.ascontiguousarray(value)
    dtype = value.dtype.name
    if dtype == "bfloat16":
        return torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
    if dtype in ("float8_e5m2", "float8_e4m3fn"):
        return torch.from_numpy(value.view(np.uint8)).view(getattr(torch, dtype))
    if dtype == "hifloat8":
        # PyTorch has no hifloat8 scalar type; TTK decodes these output bytes
        # using the dtype in the API/case metadata.
        return torch.from_numpy(value.view(np.int8))
    return torch.from_numpy(value)


def _quant_dtype(dst_type, output_dtypes=None, output=None):
    if output is not None:
        dtype = str(output.dtype).removeprefix("torch.")
    elif output_dtypes is not None:
        dtype = str(output_dtypes[0]).removeprefix("torch.")
    else:
        dtype = _DST_TYPES[dst_type]
    if dtype not in _NUMPY_TYPES:
        raise ValueError(f"Unsupported quantized output dtype: {dtype}")
    return dtype


def _quantize(value, scale, zero_point, div_mode, dtype):
    scale = np.asarray(scale, dtype=np.float32)
    value = value / scale if div_mode else value * scale
    if zero_point is not None:
        value = value + np.asarray(zero_point, dtype=np.float32)
    low, high = _SATURATION[dtype]
    value = np.nan_to_num(value, nan=0.0, neginf=low, posinf=high)
    if dtype == "int8":
        value = np.rint(value)
    return np.clip(value, low, high).astype(_NUMPY_TYPES[dtype])


def _reference(
    x1,
    x2,
    gamma,
    scales1,
    scales2,
    zero_points1,
    zero_points2,
    bias,
    epsilon,
    div_mode,
    dtype,
    output_res=False,
):
    x_sum = x1.astype(np.float32) + x2.astype(np.float32)
    if x_sum.size:
        # gamma describes the entire normalized suffix; axis is quantization's
        # axis and does not restrict RMS normalization to the final dimension.
        norm_rank = max(gamma.ndim, 1)
        axes = tuple(range(x_sum.ndim - norm_rank, x_sum.ndim))
        variance = np.mean(x_sum * x_sum, axis=axes, keepdims=True)
        normalized = x_sum / np.sqrt(variance + np.float32(epsilon))
        normalized = (normalized * gamma.astype(np.float32)).reshape(x1.shape)
    else:
        normalized = x_sum.copy()
    residual = normalized.astype(x1.dtype) if output_res else None
    value = normalized if bias is None else normalized + bias.astype(np.float32)
    y1 = _quantize(value, scales1, zero_points1, div_mode, dtype).reshape(x1.shape)
    y2 = None
    if scales2 is not None or zero_points2 is not None:
        y2 = _quantize(
            value, 1.0 if scales2 is None else scales2, zero_points2, div_mode, dtype
        ).reshape(x1.shape)
    return [y1, y2, x_sum.astype(x1.dtype), residual]


def _torch_reference(
    x1,
    x2,
    gamma,
    scales1,
    scales2,
    zero_points1,
    zero_points2,
    bias,
    epsilon,
    div_mode,
    dtype,
    output_res=False,
):
    # An independent mathematical reference: no NumPy normalization or golden
    # call. Only the final storage encoding is shared for hifloat8/FP8.
    arrays = [
        _numpy(v)
        for v in (x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, bias)
    ]
    x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, bias = arrays

    def to_double(a):
        return torch.from_numpy(a.astype(np.float64))

    x_sum = to_double(x1) + to_double(x2)
    axes = tuple(range(x_sum.ndim - max(gamma.ndim, 1), x_sum.ndim))
    if x_sum.numel():
        normalized = x_sum * torch.rsqrt(
            torch.mean(torch.square(x_sum), dim=axes, keepdim=True) + epsilon
        )
        normalized = (normalized * to_double(gamma)).reshape(x1.shape)
    else:
        normalized = x_sum.clone()
    residual = _torch(normalized.numpy().astype(x1.dtype)) if output_res else None
    value = normalized if bias is None else normalized + to_double(bias)

    def quant(scale, zero_point):
        scaled = value / to_double(scale) if div_mode else value * to_double(scale)
        if zero_point is not None:
            scaled = scaled + to_double(zero_point)
        low, high = _SATURATION[dtype]
        scaled = torch.nan_to_num(scaled, nan=0.0, neginf=low, posinf=high)
        if dtype == "int8":
            scaled = torch.round(scaled)
        encoded = scaled.clamp(low, high).numpy().astype(_NUMPY_TYPES[dtype])
        encoded = encoded.reshape(x1.shape)
        return encoded if dtype == "hifloat8" else _torch(encoded)

    y2 = None
    if scales2 is not None or zero_points2 is not None:
        y2 = quant(np.array(1.0) if scales2 is None else scales2, zero_points2)
    return [
        quant(scales1, zero_points1),
        y2,
        _torch(x_sum.numpy().astype(x1.dtype)),
        residual,
    ]


def add_rms_norm_quant(
    x1,
    x2,
    gamma,
    scales1,
    scales2=None,
    zero_points1=None,
    zero_points2=None,
    beta=None,
    axis=-1,
    epsilon=1e-6,
    div_mode=True,
    dst_type=2,
    output_dtypes=None,
    **kwargs,
):
    dtype = _quant_dtype(dst_type, output_dtypes)
    return _reference(
        x1,
        x2,
        gamma,
        scales1,
        scales2,
        zero_points1,
        zero_points2,
        beta,
        epsilon,
        div_mode,
        dtype,
    )[:3]


def aclnn_add_rms_norm_quant(
    x1,
    x2,
    gamma,
    scales1,
    scales2_optional=None,
    zero_points1_optional=None,
    zero_points2_optional=None,
    axis=-1,
    epsilon=1e-6,
    div_mode=True,
    y1_out=None,
    y2_out=None,
    x_out=None,
    dst_type=2,
    **kwargs,
):
    values = [
        _numpy(v)
        for v in (
            x1,
            x2,
            gamma,
            scales1,
            scales2_optional,
            zero_points1_optional,
            zero_points2_optional,
        )
    ]
    dtype = _quant_dtype(dst_type, output=y1_out)
    return [_torch(v) for v in _reference(*values, None, epsilon, div_mode, dtype)[:3]]


class TorchNativeReference:
    def __call__(
        self,
        x1,
        x2,
        gamma,
        scales1,
        scales2=None,
        zero_points1=None,
        zero_points2=None,
        beta=None,
        axis=-1,
        epsilon=1e-6,
        div_mode=True,
        dst_type=2,
        **kwargs,
    ):
        return _torch_reference(
            x1,
            x2,
            gamma,
            scales1,
            scales2,
            zero_points1,
            zero_points2,
            beta,
            epsilon,
            div_mode,
            _DST_TYPES[dst_type],
        )[:3]


# The third-party ACLNN route must forward output dtype metadata when output
# tensors are omitted from the provider arguments.
class TorchAclnnReference:
    def __call__(
        self,
        x1,
        x2,
        gamma,
        scales1,
        scales2_optional=None,
        zero_points1_optional=None,
        zero_points2_optional=None,
        *,
        axis=-1,
        epsilon=1e-6,
        div_mode=True,
        y1_out=None,
        y2_out=None,
        x_out=None,
        dst_type=None,
        output_dtypes=None,
        output_shapes=None,
        **kwargs,
    ):
        if y1_out is None and dst_type is None and output_dtypes is None:
            raise ValueError(
                "ACLNN third_party requires output dtype metadata; "
                "this TTK XPU route must forward y1_out or dst_type"
            )
        return _torch_reference(
            x1,
            x2,
            gamma,
            scales1,
            scales2_optional,
            zero_points1_optional,
            zero_points2_optional,
            None,
            epsilon,
            div_mode,
            _quant_dtype(dst_type, output_dtypes, y1_out),
        )[:3]


class AddRmsNormQuantTestSpec:
    golden = staticmethod(add_rms_norm_quant)
    third_party = {"torch": TorchNativeReference}
    tolerance = _TOLERANCE


class AclnnAddRmsNormQuantTestSpec:
    golden = staticmethod(aclnn_add_rms_norm_quant)
    third_party = {"torch": TorchAclnnReference}
    tolerance = _TOLERANCE
