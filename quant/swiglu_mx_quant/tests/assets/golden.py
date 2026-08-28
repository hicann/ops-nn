#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK custom golden for swiglu_mx_quant (SwiGLU + DynamicMxQuant fusion operator).

Inputs (positional, in op-def order):
    x           : numpy array (fp16/bf16)
    group_index : numpy int64 array, OPTIONAL (absent -> None)

Attributes (via **kwargs, from CSV `attributes`):
    activate_dim   : int   (default -1)   SwiGLU split axis
    activate_left  : bool  (default False) True=left half is gate
    swiglu_mode    : int   (default 0)    0=SwiGLU, 1=interleaved clamp, 2=split clamp, 3=split sigmoid-clamp
    clamp_limit    : float (default 7.0)  clamp bound for mode 1/2/3
    glu_alpha      : float (default 1.702) sigmoid scale for mode 1/2
    glu_bias       : float (default 1.0)  bias added to linear path for mode 1/2
    group_mode     : int   (default 0)
    axis           : int   (default -1)   quantization axis
    dst_type       : int   (default 40)   40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2
    round_mode     : str   (default "rint")
    scale_alg      : int   (default 0)    0=per-blockscale, 1=per-block FP8
    max_dtype_value: float (default 0.0)
    block_size     : int   (fixed 32)

Outputs:
    y       : quantized result (dst_type)
    mxscale : scale factors (FP8_E8M0)

Reference (mirrors docs/aclnnSwigluMxQuant.md and kernel ComputeVfSwigluV1-V4):

    mode 0 (SwiGLU):
        chunk x along activate_dim into [A, B]
        y = silu(A) * B  (if activate_left: silu(A)*B, else silu(B)*A)

    mode 1 (interleaved clamp):
        A = x[..., ::2], B = x[..., 1::2]
        A = clamp(A, max=clamp_limit)
        B = clamp(B, -clamp_limit, clamp_limit)
        y = A * sigmoid(glu_alpha * A) * (B + glu_bias)

    mode 2 (split clamp):
        chunk x along activate_dim into [x_glu, x_linear]
        x_glu = clamp(x_glu, max=clamp_limit)
        x_linear = clamp(x_linear, -clamp_limit, clamp_limit)
        y = x_glu * sigmoid(glu_alpha * x_glu) * (x_linear + glu_bias)

    mode 3 (split sigmoid-then-clamp):
        chunk x along activate_dim into [x_glu, x_linear]
        x_glu = x_glu * sigmoid(x_glu)           # alpha=1, no bias
        x_glu = clamp(x_glu, max=clamp_limit)
        x_linear = clamp(x_linear, -clamp_limit, clamp_limit)
        y = x_glu * x_linear
"""

import math
import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

try:
    from ml_dtypes import float8_e4m3fn as _fp8_e4m3
    from ml_dtypes import float8_e5m2 as _fp8_e5m2
    from ml_dtypes import float8_e8m0 as _fp8_e8m0
except ImportError:
    _fp8_e4m3 = None
    _fp8_e5m2 = None
    _fp8_e8m0 = None

try:
    from ml_dtypes import float4_e2m1_fn as _fp4_e2m1
    from ml_dtypes import float4_e1m2_fn as _fp4_e1m2
except ImportError:
    _fp4_e2m1 = None
    _fp4_e1m2 = None


_DST_TYPE_MAP = {
    40: ("float4_e2m1", _fp4_e2m1, 2),
    41: ("float4_e1m2", _fp4_e1m2, 0),
    36: ("float8_e4m3fn", _fp8_e4m3, 8),
    35: ("float8_e5m2", _fp8_e5m2, 15),
}


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p


def _sigmoid(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _swiglu(
    x_fp32, dim_pos, swiglu_mode, activate_left, clamp_limit, glu_alpha, glu_bias
):
    """Compute SwiGLU activation (modes 0-3), return fp32 result."""
    pre = _prod(x_fp32.shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(x_fp32.shape[dim_pos:])
    xf = x_fp32.reshape(pre, cut).astype(np.float32)
    h = cut // 2

    if swiglu_mode == 0:
        a = xf[:, :h]
        b = xf[:, h:]
        if activate_left:
            res = _sigmoid(a) * a * b
        else:
            res = _sigmoid(b) * a * b
    elif swiglu_mode == 1:
        a = xf[:, 0::2]
        b = xf[:, 1::2]
        a = np.clip(a, None, clamp_limit)
        b = np.clip(b, -clamp_limit, clamp_limit)
        res = a * _sigmoid(glu_alpha * a) * (b + glu_bias)
    elif swiglu_mode == 2:
        a = xf[:, :h]
        b = xf[:, h:]
        if not activate_left:
            a, b = b, a
        a = np.clip(a, None, clamp_limit)
        b = np.clip(b, -clamp_limit, clamp_limit)
        res = a * _sigmoid(glu_alpha * a) * (b + glu_bias)
    elif swiglu_mode == 3:
        a = xf[:, :h]
        b = xf[:, h:]
        if not activate_left:
            a, b = b, a
        a = a * _sigmoid(a)
        a = np.clip(a, None, clamp_limit)
        b = np.clip(b, -clamp_limit, clamp_limit)
        res = a * b
    else:
        raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")

    y = np.zeros((pre, h), dtype=np.float32)
    y[:] = res.astype(np.float32)
    out_shape = list(x_fp32.shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    y = y.reshape(out_shape)
    return y


def _mx_quantize(data_fp32, axis_pos, dst_type, block_size, round_mode, scale_alg):
    """Dynamic MX quantization, returns (quantized_y, mxscale)."""
    dst_name, dst_dtype, emax = _DST_TYPE_MAP[dst_type]

    shape = list(data_fp32.shape)
    pre_q = _prod(shape[:axis_pos]) if axis_pos > 0 else 1
    q_dim = shape[axis_pos]
    post_q = _prod(shape[axis_pos + 1 :]) if axis_pos + 1 < len(shape) else 1
    flat = data_fp32.reshape(pre_q, q_dim, post_q)

    n_blocks = math.ceil(q_dim / block_size)
    y_flat = np.zeros((pre_q, q_dim, post_q), dtype=np.float32)
    scale_flat = np.zeros((pre_q, n_blocks, post_q), dtype=np.float32)

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, q_dim)
        chunk = flat[:, start:end, :]
        pad_len = block_size - (end - start)
        if pad_len > 0:
            chunk = np.pad(chunk, ((0, 0), (0, pad_len), (0, 0)), mode="constant")

        abs_max = np.max(np.abs(chunk), axis=1, keepdims=True)
        abs_max = np.where(abs_max == 0, 1.0, abs_max)

        shared_exp = np.floor(np.log2(abs_max)) - emax
        shared_exp = np.where(shared_exp < 0, 0, shared_exp)
        mxscale = np.power(2.0, shared_exp.astype(np.int32).astype(np.float32))

        scaled = chunk / np.where(mxscale == 0, 1.0, mxscale)

        if round_mode == "floor":
            scaled_q = np.floor(scaled)
        elif round_mode == "round":
            scaled_q = np.round(scaled)
        else:
            scaled_q = np.rint(scaled)

        actual_len = end - start
        y_flat[:, start:end, :] = scaled_q[:, :actual_len, :]
        scale_flat[:, b, :] = mxscale[:, 0, :]

    y_out = y_flat.reshape(shape)
    scale_shape = list(shape)
    scale_shape[axis_pos] = n_blocks
    scale_shape.append(2)
    scale_out = np.zeros(scale_shape, dtype=np.float32)

    if post_q == 1:
        scale_out[:, :, 0, 0] = scale_flat[:, :, 0]
        scale_out[:, :, 1, 0] = scale_flat[:, :, 0]
    else:
        for i in range(2):
            scale_out[:, :, :, i] = scale_flat[:, :, :]

    return y_out, scale_out, dst_dtype


def __golden_swiglu_mx_quant(*input_arrays, **kwargs):
    x = np.asarray(input_arrays[0])
    group_index = None
    if len(input_arrays) > 1 and input_arrays[1] is not None:
        group_index = np.asarray(input_arrays[1])

    activate_dim = int(kwargs.get("activate_dim", -1))
    activate_left = bool(kwargs.get("activate_left", False))
    swiglu_mode = int(kwargs.get("swiglu_mode", 0))
    clamp_limit = float(kwargs.get("clamp_limit", 7.0))
    glu_alpha = float(kwargs.get("glu_alpha", 1.702))
    glu_bias = float(kwargs.get("glu_bias", 1.0))
    axis = int(kwargs.get("axis", -1))
    dst_type = int(kwargs.get("dst_type", 40))
    round_mode = str(kwargs.get("round_mode", "rint"))
    scale_alg = int(kwargs.get("scale_alg", 0))
    block_size = 32

    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes is not None and len(output_dtypes) > 0:
        pass

    ndim = x.ndim
    dim_pos = activate_dim % ndim
    axis_pos = axis % ndim

    if "bfloat16" in str(x.dtype):
        x_fp32 = x.astype(np.float32)
    elif "float16" in str(x.dtype):
        x_fp32 = x.astype(np.float32)
    else:
        x_fp32 = x.astype(np.float32)

    swiglu_result = _swiglu(
        x_fp32, dim_pos, swiglu_mode, activate_left, clamp_limit, glu_alpha, glu_bias
    )

    if "bfloat16" in str(x.dtype):
        swiglu_result = (
            swiglu_result.astype(_bf16) if _bf16 is not None else swiglu_result
        )
    elif "float16" in str(x.dtype):
        swiglu_result = swiglu_result.astype(np.float16)

    swiglu_fp32 = swiglu_result.astype(np.float32)

    if group_index is not None:
        y_shape = list(swiglu_fp32.shape)
        scale_shape = list(swiglu_fp32.shape)
        scale_shape[axis_pos] = math.ceil(scale_shape[axis_pos] / block_size)
        scale_shape.append(2)

        _, dst_dtype, _ = _DST_TYPE_MAP[dst_type]
        if dst_dtype is not None:
            y = np.zeros(y_shape, dtype=dst_dtype)
        else:
            y = np.zeros(y_shape, dtype=np.float32)
        scale = np.zeros(
            scale_shape, dtype=_fp8_e8m0 if _fp8_e8m0 is not None else np.float32
        )

        start = 0
        for gv in group_index:
            gv = int(gv)
            y_part, scale_part, dst_dt = _mx_quantize(
                swiglu_fp32[start : start + gv],
                axis_pos,
                dst_type,
                block_size,
                round_mode,
                scale_alg,
            )
            if dst_dtype is not None:
                y[start : start + gv] = y_part.astype(dst_dtype)
            else:
                y[start : start + gv] = y_part
            scale[start : start + gv] = scale_part.astype(
                _fp8_e8m0 if _fp8_e8m0 is not None else np.float32
            )
            start += gv
    else:
        y_np, scale_np, dst_dtype = _mx_quantize(
            swiglu_fp32, axis_pos, dst_type, block_size, round_mode, scale_alg
        )
        if dst_dtype is not None:
            y = y_np.astype(dst_dtype)
        else:
            y = y_np
        scale = scale_np.astype(_fp8_e8m0 if _fp8_e8m0 is not None else np.float32)

    y = np.nan_to_num(y, nan=0.0)
    scale = np.nan_to_num(scale, nan=0.0)
    return [y, scale]


__golden__ = {"kernel": {"swiglu_mx_quant": "__golden_swiglu_mx_quant"}}
