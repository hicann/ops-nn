#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU reference for the Kernel and GEIR test paths.

The confirmed reference domain contains finite values whose rounded quantized
result fits in int8. NaN, Inf, and out-of-range conversion semantics remain
outside this TestSpec. Besides the public per-channel division mode, this
reference covers the internal multiplication modes and the legacy behavior
where other axis values fall back to per-channel division.
"""

import math

import numpy as np
import torch
import torch.nn.functional as functional


__spec__ = {
    "QuantizeAddLayerNorm": "QuantizeAddLayerNormTestSpec",
    "quantize_add_layer_norm": "QuantizeAddLayerNormTestSpec",
}

_MAIN_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_AXIS_PER_CHANNEL_MUL = -65535
_AXIS_PER_TENSOR_MUL = 65535
_QUANT_DTYPES = {
    torch.float16: (torch.float32,),
    torch.float32: (torch.float32,),
    torch.bfloat16: (torch.bfloat16, torch.float32),
}


def _matches_gamma_layout(shape, x_shape, gamma_shape):
    if shape == gamma_shape:
        return True
    if len(shape) != len(x_shape) or len(shape) < len(gamma_shape):
        return False
    leading_dims = shape[: len(shape) - len(gamma_shape)]
    return (
        all(dim == 1 for dim in leading_dims)
        and shape[-len(gamma_shape) :] == gamma_shape
    )


def _validate(
    x1,
    x2,
    gamma,
    beta,
    bias,
    scales,
    zero_points,
    dtype,
    axis,
    epsilon,
    additional_output,
    promote=False,
):
    if type(dtype) is not int or dtype != 2:
        raise ValueError("dtype must be DT_INT8(2)")
    if type(axis) is not int or not -(2**63) <= axis < 2**63:
        raise ValueError("axis must be int64")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be finite and positive")
    if type(additional_output) is not bool:
        raise ValueError("additional_output must be bool")

    required = (x1, x2, gamma, beta, bias, scales)
    if any(not isinstance(tensor, torch.Tensor) for tensor in required):
        raise TypeError("all required inputs must be tensors")
    tensors = required + (() if zero_points is None else (zero_points,))
    if any(
        tensor.device != x1.device or not torch.isfinite(tensor).all().item()
        for tensor in tensors
    ):
        raise ValueError("inputs must be finite tensors on one device")

    x_shape = tuple(x1.shape)
    gamma_shape = tuple(gamma.shape)
    if (
        not x_shape
        or len(gamma_shape) != 1
        or any(dim <= 0 for dim in x_shape + gamma_shape)
    ):
        raise ValueError("x1 must be nonempty and gamma must be nonempty rank 1")
    if (
        tuple(x2.shape) != x_shape
        or gamma_shape[0] != x_shape[-1]
        or tuple(beta.shape) != gamma_shape
    ):
        raise ValueError("x2, gamma, or beta shape mismatch")
    if tuple(bias.shape) not in (gamma_shape, x_shape):
        raise ValueError("bias must match gamma or x1")
    if any(tensor.dtype != x1.dtype for tensor in (x2, gamma, beta, bias)):
        raise TypeError("x1, x2, gamma, beta, and bias must have the same dtype")

    quant_params = (scales,) if zero_points is None else (scales, zero_points)
    if promote:
        if x1.dtype not in (torch.float32, torch.float64) or any(
            tensor.dtype not in (torch.float32, torch.float64)
            for tensor in quant_params
        ):
            raise TypeError("unexpected promoted dtype")
    elif x1.dtype not in _MAIN_DTYPES or any(
        tensor.dtype not in _QUANT_DTYPES[x1.dtype] for tensor in quant_params
    ):
        raise TypeError("unsupported main and quantization dtype combination")
    if zero_points is not None and zero_points.dtype != scales.dtype:
        raise TypeError("zero_points dtype must match scales dtype")

    if axis == _AXIS_PER_TENSOR_MUL:
        if any(
            tensor.numel() != 1 and tuple(tensor.shape) != gamma_shape
            for tensor in quant_params
        ):
            raise ValueError(
                "per-tensor quantization parameters must contain one element or match gamma"
            )
    elif any(
        not _matches_gamma_layout(tuple(tensor.shape), x_shape, gamma_shape)
        for tensor in quant_params
    ):
        raise ValueError(
            "per-channel quantization parameters must match gamma or use leading singleton dimensions"
        )
    if (
        axis not in (_AXIS_PER_CHANNEL_MUL, _AXIS_PER_TENSOR_MUL)
        and torch.any(scales == 0).item()
    ):
        raise ValueError("division scales must be nonzero")


def _compute(
    x1,
    x2,
    gamma,
    beta,
    bias,
    scales,
    zero_points,
    axis,
    epsilon,
    additional_output,
    promote=False,
):
    compute_dtype = x1.dtype if promote and x1.dtype == torch.float64 else torch.float32
    residual = (
        x1.contiguous().to(compute_dtype) + x2.contiguous().to(compute_dtype)
    ) + bias.to(compute_dtype)
    normalized = functional.layer_norm(
        residual,
        normalized_shape=tuple(gamma.shape),
        weight=gamma.to(compute_dtype),
        bias=beta.to(compute_dtype),
        eps=epsilon,
    )
    scale = scales.to(compute_dtype)
    if axis == _AXIS_PER_TENSOR_MUL:
        scale = scale.reshape(-1)[0]
    else:
        scale = scale.reshape(gamma.shape)
    quantized = (
        normalized * scale
        if axis in (_AXIS_PER_CHANNEL_MUL, _AXIS_PER_TENSOR_MUL)
        else normalized / scale
    )
    if zero_points is not None:
        offset = zero_points.to(compute_dtype)
        if axis == _AXIS_PER_TENSOR_MUL:
            offset = offset.reshape(-1)[0]
        else:
            offset = offset.reshape(gamma.shape)
        quantized = quantized + offset

    if any(
        not tensor.isfinite().all().item()
        for tensor in (residual, normalized, quantized)
    ):
        raise ValueError("non-finite intermediate is outside the reference domain")
    rounded = torch.round(quantized)
    if torch.any((rounded < -128) | (rounded > 127)).item():
        raise ValueError("out-of-range int8 conversion is outside the reference domain")
    return [
        rounded.to(torch.int8),
        residual.to(x1.dtype) if additional_output else None,
    ]


def _numpy_to_torch(value):
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise TypeError("Kernel/GEIR reference expects numpy.ndarray inputs")
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(value.astype(np.float32))).to(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(value))


def _torch_to_numpy(value, dtype):
    if value is None:
        return None
    if value.dtype == torch.bfloat16:
        return value.cpu().float().numpy().astype(dtype, copy=False)
    return value.cpu().numpy().astype(dtype, copy=False)


class QuantizeAddLayerNormTestSpec:
    """Shared Kernel/GEIR TestSpec with a separately executed Torch provider."""

    def golden(
        x1,
        x2,
        gamma,
        beta,
        bias,
        scales,
        zero_points=None,
        *,
        dtype,
        axis=-1,
        epsilon=1e-5,
        additional_output=False,
        **kwargs,
    ):
        inputs = [
            _numpy_to_torch(value)
            for value in (x1, x2, gamma, beta, bias, scales, zero_points)
        ]
        promote = kwargs.get("golden_mode", "Enable") == "Promote"
        _validate(*inputs, dtype, axis, epsilon, additional_output, promote=promote)
        outputs = _compute(*inputs, axis, epsilon, additional_output, promote=promote)
        return [
            _torch_to_numpy(outputs[0], np.int8),
            _torch_to_numpy(outputs[1], x1.dtype),
        ]

    class ThirdPartyImpl:
        def __init__(
            self, *, dtype, axis=-1, epsilon=1e-5, additional_output=False, **kwargs
        ):
            del kwargs
            self.dtype = dtype
            self.axis = axis
            self.epsilon = epsilon
            self.additional_output = additional_output

        def __call__(
            self, x1, x2, gamma, beta, bias, scales, zero_points=None, **kwargs
        ):
            del kwargs
            _validate(
                x1,
                x2,
                gamma,
                beta,
                bias,
                scales,
                zero_points,
                self.dtype,
                self.axis,
                self.epsilon,
                self.additional_output,
            )
            residual = torch.add(x1.to(torch.float32), x2.to(torch.float32)) + bias.to(
                torch.float32
            )
            normalized = functional.layer_norm(
                residual,
                normalized_shape=tuple(gamma.shape),
                weight=gamma.to(torch.float32),
                bias=beta.to(torch.float32),
                eps=self.epsilon,
            )
            scale = scales.to(torch.float32)
            if self.axis == _AXIS_PER_TENSOR_MUL:
                scale = scale.reshape(-1)[0]
            else:
                scale = scale.reshape(gamma.shape)
            offset = None if zero_points is None else zero_points.to(torch.float32)
            if offset is not None:
                if self.axis == _AXIS_PER_TENSOR_MUL:
                    offset = offset.reshape(-1)[0]
                else:
                    offset = offset.reshape(gamma.shape)
            quantized = (
                normalized * scale
                if self.axis in (_AXIS_PER_CHANNEL_MUL, _AXIS_PER_TENSOR_MUL)
                else normalized / scale
            )
            if offset is not None:
                quantized = quantized + offset
            if any(
                not tensor.isfinite().all().item()
                for tensor in (residual, normalized, quantized)
            ):
                raise ValueError("non-finite third-party result")
            rounded = torch.round(quantized)
            if torch.any((rounded < -128) | (rounded > 127)).item():
                raise ValueError("out-of-range third-party int8 result")
            return [
                rounded.to(torch.int8),
                residual.to(x1.dtype) if self.additional_output else None,
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "int8": {"standard": "quant"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
