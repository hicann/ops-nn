# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# /
"""CPU references for the AddRmsNormDynamicQuant V1/V2 family on Ascend 950.

Adapted from MyTestCase/AddRmsNormDynamicQuant/golden/main.py. All defined
outputs participate in comparison. Disabled output slots return None. ACLNN
INT32 outputs store eight signed INT4 values per word, low nibble first.

The residual and RMS normalization retain FP32 intermediates. Native V2's y3
and y4 are the normalization result before beta; beta affects quantization.
The NumPy implementation and the independent PyTorch FP64 implementation
share only the final storage encoding and the interface adapters.
"""

from functools import wraps

import en_dtypes
import ml_dtypes
import numpy as np


__spec__ = {
    "add_rms_norm_dynamic_quant": "NativeV1TestSpec",
    "add_rms_norm_dynamic_quant_v2": "NativeV2TestSpec",
    "aclnnAddRmsNormDynamicQuant": "AclnnV1TestSpec",
    "aclnnAddRmsNormDynamicQuantV2": "AclnnV2TestSpec",
    "torch_npu.npu_add_rms_norm_dynamic_quant": "TorchTestSpec",
    "torch.ops.npu.npu_add_rms_norm_dynamic_quant": "TorchTestSpec",
}

_DTYPES = {
    "int8": np.int8,
    "int4": ml_dtypes.int4,
    "int32": np.int32,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "hifloat8": en_dtypes.hifloat8,
}
_MAXIMUM = {
    "int8": 127.0,
    "int4": 7.0,
    "int32": 7.0,
    "float8_e4m3fn": 448.0,
    "float8_e5m2": 57344.0,
    "hifloat8": 32768.0,
}
_DST_TYPES = {
    2: "int8",
    29: "int4",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
}
_TOLERANCE = {
    "int8": {"standard": "quant"},
    "int4": {"standard": "quant"},
    "int32": {"standard": "quant", "packed_dtype": "int4"},
    "float8_e4m3fn": {"standard": "requant"},
    "float8_e5m2": {"standard": "requant"},
    "hifloat8": {"standard": "requant"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
}


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _dtype_name(value):
    try:
        return np.dtype(value).name
    except (TypeError, ValueError):
        return str(value).split(".")[-1]


def _input_dtype(value, metadata):
    dtypes = metadata.get("input_dtypes") or metadata.get("tensor_dtypes")
    name = _dtype_name(dtypes[0] if dtypes else value.dtype)
    return ml_dtypes.bfloat16 if name == "bfloat16" else np.dtype(name)


def _encode_quantized(value, dtype):
    if dtype == "int32":
        if value.shape[-1] % 8:
            raise ValueError("Packed INT4 requires a last dimension divisible by 8")
        nibbles = (value.astype(np.int32) & 15).astype(np.uint32)
        nibbles = nibbles.reshape(*value.shape[:-1], value.shape[-1] // 8, 8)
        words = np.bitwise_or.reduce(
            nibbles << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
        )
        return words.view(np.int32)
    return value.astype(_DTYPES[dtype])


def _numpy_reference(
    x1, x2, gamma, smooth1, smooth2, beta, epsilon, flags, dtype, input_dtype
):
    residual = _array(x1) + _array(x2)
    if residual.size == 0:
        normalized = np.empty_like(residual, dtype=np.float32)
    else:
        rstd = np.reciprocal(
            np.sqrt(
                np.mean(residual * residual, axis=-1, keepdims=True)
                + np.float32(epsilon)
            )
        )
        normalized = residual * rstd * _array(gamma)
    value = normalized if beta is None else normalized + _array(beta)
    quantized = []
    scales = []
    for enabled, smooth in zip(flags[:2], (smooth1, smooth2)):
        if not enabled:
            quantized.append(None)
            scales.append(None)
            continue
        if residual.size == 0:
            quantized.append(_encode_quantized(value, dtype))
            # ACLNN's empty shortcut does not initialize nonempty scale buffers.
            scales.append(None)
            continue
        smoothed = value if smooth is None else value * _array(smooth)
        scale = np.max(np.abs(smoothed), axis=-1, keepdims=True) / np.float32(
            _MAXIMUM[dtype]
        )
        q = smoothed / np.where(scale == 0, np.float32(1), scale)
        if dtype in ("int8", "int4", "int32"):
            q = np.clip(np.rint(q), -_MAXIMUM[dtype] - 1, _MAXIMUM[dtype])
        quantized.append(_encode_quantized(q, dtype))
        scales.append(scale.squeeze(-1).astype(np.float32))
    outputs = [*quantized, residual.astype(input_dtype), *scales]
    return outputs, normalized, normalized.astype(input_dtype)


def _torch_reference(
    x1, x2, gamma, smooth1, smooth2, beta, epsilon, flags, dtype, input_dtype
):
    import torch

    def tensor(value):
        if hasattr(value, "detach"):
            return value.detach().to(device="cpu", dtype=torch.float64)
        return torch.from_numpy(np.asarray(value).astype(np.float64))

    residual = tensor(x1) + tensor(x2)
    if residual.numel() == 0:
        normalized = torch.empty_like(residual)
    else:
        energy = (
            torch.sum(torch.square(residual), dim=-1, keepdim=True) / residual.shape[-1]
        )
        normalized = (
            residual / torch.sqrt(energy + float(np.float32(epsilon))) * tensor(gamma)
        )
    value = normalized if beta is None else normalized + tensor(beta)
    quantized = []
    scales = []
    for enabled, smooth in zip(flags[:2], (smooth1, smooth2)):
        if not enabled:
            quantized.append(None)
            scales.append(None)
            continue
        if residual.numel() == 0:
            quantized.append(_encode_quantized(value.numpy(), dtype))
            scales.append(None)
            continue
        smoothed = value if smooth is None else value * tensor(smooth)
        scale = torch.amax(torch.abs(smoothed), dim=-1, keepdim=True) / _MAXIMUM[dtype]
        q = smoothed / torch.where(scale == 0, torch.ones_like(scale), scale)
        if dtype in ("int8", "int4", "int32"):
            q = torch.round(q).clamp(-_MAXIMUM[dtype] - 1, _MAXIMUM[dtype])
        quantized.append(_encode_quantized(q.numpy(), dtype))
        scales.append(scale.squeeze(-1).numpy().astype(np.float32))
    norm_array = normalized.numpy()
    outputs = [*quantized, residual.numpy().astype(input_dtype), *scales]
    return outputs, norm_array.astype(np.float32), norm_array.astype(input_dtype)


def _reference(metadata):
    return (
        _torch_reference if metadata.get("_independent_reference") else _numpy_reference
    )


def _flags(mask, smooth1, smooth2, count):
    if mask is not None and len(mask):
        if len(mask) != count:
            raise ValueError(f"output mask must contain {count} flags")
        return [bool(value) for value in mask]
    return [True, smooth1 is not None and smooth2 is not None] + [True] * (count - 2)


def native_v1(
    x1,
    x2,
    gamma,
    smooth_scale1=None,
    smooth_scale2=None,
    beta=None,
    epsilon=1e-6,
    output_mask=None,
    dst_type=2,
    **kwargs,
):
    flags = _flags(output_mask, smooth_scale1, smooth_scale2, 2)
    return _reference(kwargs)(
        x1,
        x2,
        gamma,
        smooth_scale1,
        smooth_scale2,
        beta,
        epsilon,
        flags,
        _DST_TYPES[dst_type],
        _input_dtype(x1, kwargs),
    )[0]


def native_v2(
    x1,
    x2,
    gamma,
    smooth_scale1=None,
    smooth_scale2=None,
    beta=None,
    epsilon=1e-6,
    output_mask=None,
    dst_type=2,
    **kwargs,
):
    flags = _flags(output_mask, smooth_scale1, smooth_scale2, 4)
    base, norm32, norm_input = _reference(kwargs)(
        x1,
        x2,
        gamma,
        smooth_scale1,
        smooth_scale2,
        beta,
        epsilon,
        flags,
        _DST_TYPES[dst_type],
        _input_dtype(x1, kwargs),
    )
    return (
        base[:2]
        + [norm32 if flags[2] else None, norm_input if flags[3] else None]
        + base[2:]
    )


def aclnn_v1(
    x1,
    x2,
    gamma,
    smooth_scale1_optional=None,
    smooth_scale2_optional=None,
    epsilon=1e-6,
    y1_out=None,
    y2_out=None,
    x_out=None,
    scale1_out=None,
    scale2_out=None,
    **kwargs,
):
    if _array(x1).ndim not in range(2, 9):
        # Negative-rank cases require an actual API rejection, never numeric PASS.
        return [None] * 5
    dtypes = kwargs.get("tensor_dtypes")
    dtype = _dtype_name(dtypes[5] if dtypes else y1_out.dtype)
    return _reference(kwargs)(
        x1,
        x2,
        gamma,
        smooth_scale1_optional,
        smooth_scale2_optional,
        None,
        epsilon,
        [True, smooth_scale2_optional is not None],
        dtype,
        _input_dtype(x1, kwargs),
    )[0]


def aclnn_v2(
    x1,
    x2,
    gamma,
    smooth_scale1_optional=None,
    smooth_scale2_optional=None,
    beta_optional=None,
    epsilon=1e-6,
    output_mask=None,
    y1_out=None,
    y2_out=None,
    x_out=None,
    scale1_out=None,
    scale2_out=None,
    **kwargs,
):
    if _array(x1).ndim not in range(2, 9):
        return [None] * 5
    dtypes = kwargs.get("tensor_dtypes")
    dtype = _dtype_name(dtypes[6] if dtypes else y1_out.dtype)
    flags = _flags(output_mask, smooth_scale1_optional, smooth_scale2_optional, 2)
    return _reference(kwargs)(
        x1,
        x2,
        gamma,
        smooth_scale1_optional,
        smooth_scale2_optional,
        beta_optional,
        epsilon,
        flags,
        dtype,
        _input_dtype(x1, kwargs),
    )[0]


def torch_api(
    x1,
    x2,
    gamma,
    *,
    smooth_scale1=None,
    smooth_scale2=None,
    beta=None,
    epsilon=1e-6,
    output_mask=None,
    y_dtype=None,
    **kwargs,
):
    dtype = "int8" if y_dtype is None else _dtype_name(y_dtype)
    if dtype == "quint4x2":
        dtype = "int32"
    flags = _flags(output_mask, smooth_scale1, smooth_scale2, 2)
    return _reference(kwargs)(
        x1,
        x2,
        gamma,
        smooth_scale1,
        smooth_scale2,
        beta,
        epsilon,
        flags,
        dtype,
        _input_dtype(x1, kwargs),
    )[0]


def _independent(function):
    @wraps(function)
    def reference(*args, **kwargs):
        kwargs["_independent_reference"] = True
        return function(*args, **kwargs)

    return reference


class NativeV1TestSpec:
    golden = native_v1
    tolerance = _TOLERANCE
    third_party = {"torch": _independent(native_v1)}


class NativeV2TestSpec:
    golden = native_v2
    tolerance = _TOLERANCE
    third_party = {"torch": _independent(native_v2)}


class AclnnV1TestSpec:
    golden = aclnn_v1
    tolerance = _TOLERANCE

    class TorchReference:
        def __call__(
            self,
            x1,
            x2,
            gamma,
            smoothScale1Optional=None,
            smoothScale2Optional=None,
            epsilon=1e-6,
            tensor_dtypes=None,
        ):
            return aclnn_v1(
                x1,
                x2,
                gamma,
                smoothScale1Optional,
                smoothScale2Optional,
                epsilon,
                tensor_dtypes=tensor_dtypes,
                _independent_reference=True,
            )

    third_party = {"torch": TorchReference}


class AclnnV2TestSpec:
    golden = aclnn_v2
    tolerance = _TOLERANCE

    class TorchReference:
        def __call__(
            self,
            x1,
            x2,
            gamma,
            smoothScale1Optional=None,
            smoothScale2Optional=None,
            betaOptional=None,
            epsilon=1e-6,
            outputMask=None,
            tensor_dtypes=None,
        ):
            return aclnn_v2(
                x1,
                x2,
                gamma,
                smoothScale1Optional,
                smoothScale2Optional,
                betaOptional,
                epsilon,
                outputMask,
                tensor_dtypes=tensor_dtypes,
                _independent_reference=True,
            )

    third_party = {"torch": TorchReference}


class TorchTestSpec:
    golden = torch_api
    tolerance = _TOLERANCE
    third_party = {"torch": _independent(torch_api)}
