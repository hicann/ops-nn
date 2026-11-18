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
"""Golden reference for the Selu operator.

Selu: y = scale * (max(0, x) + min(0, alpha * (exp(x) - 1))), with
alpha = 1.6732632423543772 and scale = 1.0507009873554805.

All golden() paths delegate to _benchmark_selu (oracle: torch.selu);
int32/int8 inputs are computed in float32 and cast back per device Cast
semantics (int32 saturates on overflow, int8 wraps).
"""

__spec__ = {
    "selu": "SeluKernelTestSpec",
    "Selu": "SeluKernelTestSpec",
    "aclnnSelu": "AclnnSeluTestSpec",
    "aclnnInplaceSelu": "AclnnInplaceSeluTestSpec",
    "tf.nn.selu": "SeluTensorFlowSpec",
    "tf.raw_ops.Selu": "SeluTensorFlowSpec",
    # torch path not registered: F.selu dispatches to builtin Elu on NPU
}

ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805

# Shared full-dtype tolerance table, consumed per output dtype: TTK looks a
# case's standard up by its output dtype, so keys beyond a lane's dtype face
# are never consumed. cross_check for float dtypes, binary_equal for ints,
# float64 isclose (cross_check is physically unreachable for f64).
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L2"},
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float64": {"standard": "isclose"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


def _selu_torch_composed(t):
    """SELU via basic torch ops for int32/int8 (torch.selu rejects ints):
    compute in float32, cast back per device Cast (int32 saturates, int8 wraps)."""
    import torch

    xf = t.to(torch.float32)
    yt = SCALE * (
        torch.clamp(xf, min=0.0) + torch.clamp(ALPHA * torch.expm1(xf), max=0.0)
    )
    if t.dtype is torch.int32:
        yt = _saturate_to_int32_range(yt)
    return yt.to(t.dtype)


def _saturate_to_int32_range(y):
    """Clamp to the int32 range in float64: torch ``.to(int32)`` yields
    INT_MIN on out-of-range floats while the device saturates, and float32
    cannot represent INT32_MAX."""
    import torch

    info = torch.iinfo(torch.int32)
    return y.to(torch.float64).clamp(float(info.min), float(info.max))


def _benchmark_selu(x):
    """Shared SELU benchmark (oracle: torch.selu). Normalizes the input
    carrier (torch/tf/numpy) to torch, computes, and casts back; bfloat16
    crosses the numpy border as an int16 bit view (ml_dtypes storage)."""
    import numpy as np
    import torch

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        t = x.detach().cpu()
    else:
        if type(x).__module__.split(".", 1)[0] == "tensorflow":  # tf, no import
            if x.dtype.name == "bfloat16":  # tf.DType.as_numpy_dtype = bf16 bridge
                x = np.asarray(x.numpy()).view(x.dtype.as_numpy_dtype)
            else:
                x = np.asarray(x.numpy())
        x = np.asarray(x)
        if x.dtype.name == "bfloat16":  # numpy bf16 rides torch as int16 bits
            t = torch.from_numpy(x.view("int16")).view(torch.bfloat16)
        else:
            t = torch.from_numpy(x)
    if t.dtype in (torch.int8, torch.int32):
        # torch.selu does not support integer dtypes -> torch basic-op composition
        y = _selu_torch_composed(t)
    else:
        y = torch.selu(t)
    if is_torch:
        return y
    if y.dtype == torch.bfloat16:
        import ml_dtypes

        return y.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return y.numpy()


def _pick_input(pred, *args, **kwargs):
    """Locate the operator input: first positional pred match, else the first
    present of the usual tensor kw names."""
    x = next((arg for arg in args if pred(arg)), None)
    if x is None:
        for name in ("x", "self", "selfRef", "features", "input", "input_x"):
            x = kwargs.get(name)
            if x is not None:
                break
    if x is None:
        raise ValueError("Missing Selu input: x/self/selfRef/features/input/input_x")
    return x


class _TorchSelu:
    """torch.selu wrapper (class form: the aclnn ``self`` input cannot bind
    a string API); int legs bridge via float32 per device Cast semantics."""

    def __call__(instance, *args, **kwargs):
        import torch

        x = _pick_input(lambda a: isinstance(a, torch.Tensor), *args, **kwargs)
        if x.dtype in (torch.int8, torch.int32):
            # torch.selu rejects ints -> bridge via float32, cast back per device Cast
            y = torch.selu(x.to(torch.float32))
            # device branch: non-CPU bare cast = device semantics (GPU verified)
            if x.device.type != "cpu":
                return [y.to(x.dtype)]
            if x.dtype is torch.int32:
                y = _saturate_to_int32_range(y)
            return [y.to(x.dtype)]
        return [torch.selu(x)]


class _TfSelu:
    """tf.raw_ops.Selu wrapper (float dtypes only); int legs bridge via
    float32 and cast back per device Cast."""

    def __call__(instance, *args, **kwargs):
        import tensorflow as tf

        x = _pick_input(
            lambda a: isinstance(a, (tf.Tensor, tf.Variable)), *args, **kwargs
        )
        if x.dtype in (tf.int8, tf.int32):
            y = tf.raw_ops.Selu(features=tf.cast(x, tf.float32))
            # device branch: non-CPU bare cast = device semantics (GPU verified)
            if tf.DeviceSpec.from_string(x.device).device_type != "CPU":
                return [tf.cast(y, x.dtype)]
            if x.dtype is tf.int32:
                y = tf.cast(y, tf.float64)
                y = tf.clip_by_value(y, -2147483648.0, 2147483647.0)
            else:
                # int8: |selu(x)| <= 133.44 < 135; wrap two's complement (device Cast)
                y = tf.cast(y, tf.int32)
                y = (y + 128) % 256 - 128
            return [tf.cast(y, x.dtype)]
        return [tf.raw_ops.Selu(features=x)]


# Per-path third_party: the framework-agnostic kernel/GEIR entry lists both
# wrappers (they bridge int legs); the TF spec binds the tf wrapper class so
# the bridged legs stay available to the tf entry.
_TP_TORCH = {"torch": _TorchSelu}
_TP_ALL = {**_TP_TORCH, "tf": _TfSelu}


def _golden(x=None, out=None, **kwargs):
    """Shared single-input golden; carriers normalize inside the benchmark.
    Optional ``out``: cast to its dtype when given; out stays read-only.
    ``x`` falls back to the usual kw names: TTK's raw_ops convention delivers
    tf inputs by keyword (features=...), unlike the positional kernel/aclnn."""
    if x is None:
        x = _pick_input(lambda a: True, **kwargs)
    y = _benchmark_selu(x)
    return [y] if out is None else [y.to(out.dtype)]


class SeluKernelTestSpec:
    """Kernel/GEIR spec (CSV ``op_name`` "selu"/"Selu", numpy input)."""

    golden = _golden
    third_party = _TP_ALL
    tolerance = _TOLERANCE


class AclnnSeluTestSpec:
    """ACLNN spec (CSV ``api_name`` aclnnSelu, torch input, read-only ``out``)."""

    golden = _golden
    third_party = _TP_TORCH
    tolerance = _TOLERANCE


class AclnnInplaceSeluTestSpec:
    """ACLNN inplace spec (CSV ``api_name`` aclnnInplaceSelu, torch input)."""

    golden = _golden
    third_party = _TP_TORCH
    tolerance = _TOLERANCE


class SeluTensorFlowSpec:
    """TF spec (CSV ``api_name`` "tf.nn.selu"/"tf.raw_ops.Selu", tf input)."""

    golden = _golden
    third_party = {"tf": _TfSelu}
    tolerance = _TOLERANCE
