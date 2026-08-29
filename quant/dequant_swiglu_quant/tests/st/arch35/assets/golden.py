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
TTK custom golden for dequant_swiglu_quant (arch35 / Ascend950), covering swiglu_mode 0/1/2/3.

Inputs (positional, in op-def order):
    x                : numpy array, int32 (needs dequant) or fp16/bf16 (already "dequantized")
    weight_scale     : numpy fp32 array, OPTIONAL. Simple (2H,) last-axis broadcast only
                        (grouped weight_scale (G, 2H) is NOT supported by this golden -- keep
                        test cases free of group_index when weight_scale is used).
    activation_scale : numpy fp32 array, OPTIONAL, per-row scalar, shape = x.shape[:-1] (broadcast
                        over the activation axis after it has been moved to the last position).
    bias             : numpy array, OPTIONAL. Dequant bias (NOT the glu_bias attribute). This
                        golden adds it (as float32) AFTER the weight/activation scale multiply,
                        matching the kernel's float-family-bias order. int32-bias order in the
                        real kernel differs (bias added before the scale multiply) -- avoid
                        constructing test cases that rely on this path; none of the mode2/3 cases
                        in this test round use the bias input at all.
    quant_scale      : numpy fp32 array, OPTIONAL (only used when quant_mode == 'static').
    quant_offset     : numpy fp32 array, OPTIONAL (only used when quant_mode == 'static').
    group_index      : numpy int array, OPTIONAL. NOT supported by this golden (grouped dequant is
                        out of scope for this validation round) -- omit from test cases.

Attributes (via **kwargs, from CSV `attributes`):
    activate_left : bool  (default True)   True: x_glu=front half, x_linear=back half
    quant_mode    : str   (default 'static') 'static' | 'dynamic'
    dst_type      : int   (default 2 = int8)
    round_mode    : str   (default 'rint') -- informational only, golden always uses round-half-
                    to-even (np.rint / dtype cast default) which matches CAST_RINT
    activate_dim  : int   (default -1)
    swiglu_mode   : int   (default 0)      0/1/2/3
    clamp_limit   : float (default 7.0)
    glu_alpha     : float (default 1.702)
    glu_bias      : float (default 1.0)

Output:
    y     : quantized activation output, dtype = output_dtypes[0]
    scale : fp32, per-row dynamic-quant scale (all-ones-shape zeros for static quant mode, since
            static quant does not produce a meaningful per-row scale output)

Reference math (mirrors op_kernel/arch35/dequant_swiglu_quant_common.h::SwigluSingleYWithQuantScale,
cross-checked against op_kernel/dequant_swiglu_quant.h::ComputeSwiGLU and
operators/dequant_swiglu_quant/docs/REQUIREMENTS.md):
    mode0: x_glu, x_linear = front/back split (per activate_left)
           y = x_glu * sigmoid(x_glu) * x_linear                          (no clamp/alpha/bias)
    mode1: x_glu, x_linear = interleaved split x[...,0::2] / x[...,1::2]
           x_glu = min(x_glu, L); x_linear = clip(x_linear, -L, L)
           y = x_glu * sigmoid(alpha*x_glu) * (x_linear + bias)
    mode2: x_glu, x_linear = front/back split (per activate_left)         [= mode1 math + mode0 split]
           x_glu = min(x_glu, L); x_linear = clip(x_linear, -L, L)
           y = x_glu * sigmoid(alpha*x_glu) * (x_linear + bias)
    mode3: x_glu, x_linear = front/back split (per activate_left)
           x_glu = SiLU(x_glu) [alpha=1, no pre-clamp], then x_glu = min(x_glu, L)
           x_linear = clip(x_linear, -L, L)                               (no bias)
           y = x_glu * x_linear

Dynamic quantization (per row): scale = max(|y_row|) / scalar_max(dst_type); y_q = round(y/scale)
saturate-cast to dst_type. scalar_max table (from op_kernel/arch35/dequant_swiglu_quant_static.h /
dequant_swiglu_quant_nlast.h `scalarMaxNum_`):
    int8 -> 127.0, float8_e4m3fn -> 448.0, float8_e5m2 -> 57344.0,
    float4_e2m1 -> 6.0, float4_e1m2 -> 1.75, hifloat8 -> 32768.0
"""

import numpy as np

try:
    from ttk.utilities.dtypes import (
        numpy_bfloat16,
        numpy_float4_e1m2,
        numpy_float4_e2m1,
        numpy_float8_e4m3fn,
        numpy_float8_e5m2,
    )
except ImportError:  # pragma: no cover - fall back to ml_dtypes directly
    from ml_dtypes import bfloat16 as _bf16_dtype
    from ml_dtypes import float8_e4m3fn as _f8e4m3_dtype
    from ml_dtypes import float8_e5m2 as _f8e5m2_dtype
    from ml_dtypes import float4_e2m1fn as _f4e2m1_dtype
    from ml_dtypes import float4_e1m2 as _f4e1m2_dtype

    def numpy_bfloat16():
        return _bf16_dtype

    def numpy_float8_e4m3fn():
        return _f8e4m3_dtype

    def numpy_float8_e5m2():
        return _f8e5m2_dtype

    def numpy_float4_e2m1():
        return _f4e2m1_dtype

    def numpy_float4_e1m2():
        return _f4e1m2_dtype


DT_INT8 = 2
DT_FLOAT8_E5M2 = 35
DT_FLOAT8_E4M3FN = 36
DT_FLOAT4_E2M1 = 40
DT_FLOAT4_E1M2 = 41
DT_HIFLOAT8 = 34

SCALAR_MAX_NUM = {
    DT_INT8: 127.0,
    DT_FLOAT8_E4M3FN: 448.0,
    DT_FLOAT8_E5M2: 57344.0,
    DT_FLOAT4_E2M1: 6.0,
    DT_FLOAT4_E1M2: 1.75,
    DT_HIFLOAT8: 32768.0,
}


def _sigmoid(v):
    with np.errstate(over="ignore", invalid="ignore"):
        return 1.0 / (1.0 + np.exp(-v))


def _target_np_dtype(dst_type, dtype_name):
    if dst_type == DT_INT8 or dtype_name == "int8":
        return np.int8
    if dst_type == DT_FLOAT8_E4M3FN or dtype_name == "float8_e4m3fn":
        return numpy_float8_e4m3fn()
    if dst_type == DT_FLOAT8_E5M2 or dtype_name == "float8_e5m2":
        return numpy_float8_e5m2()
    if dst_type == DT_FLOAT4_E2M1 or dtype_name == "float4_e2m1":
        return numpy_float4_e2m1()
    if dst_type == DT_FLOAT4_E1M2 or dtype_name == "float4_e1m2":
        return numpy_float4_e1m2()
    raise ValueError(
        f"Unsupported dst_type/dtype for golden: dst_type={dst_type}, dtype_name={dtype_name}"
    )


def _cast_quant(y_scaled, np_dtype):
    if np_dtype == np.int8:
        return np.clip(np.rint(y_scaled), -128, 127).astype(np.int8)
    # fp8/fp4 ml_dtypes casts already round-to-nearest-even and saturate on overflow.
    return y_scaled.astype(np.float32).astype(np_dtype)


def __golden_dequant_swiglu_quant(*input_arrays, **kwargs):
    arrays = list(input_arrays) + [None] * (7 - len(input_arrays))
    x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index = (
        arrays[:7]
    )
    if group_index is not None:
        raise NotImplementedError(
            "golden.py: grouped dequant_swiglu_quant (group_index) not supported"
        )

    x = np.asarray(x)

    activate_left = bool(kwargs.get("activate_left", True))
    quant_mode = str(kwargs.get("quant_mode", "static"))
    dst_type = int(kwargs.get("dst_type", DT_INT8))
    activate_dim = int(kwargs.get("activate_dim", -1))
    swiglu_mode = int(kwargs.get("swiglu_mode", 0))
    clamp_limit = float(kwargs.get("clamp_limit", 7.0))
    glu_alpha = float(kwargs.get("glu_alpha", 1.702))
    glu_bias = float(kwargs.get("glu_bias", 1.0))

    output_dtypes = kwargs.get("output_dtypes")
    y_dtype_name = str(output_dtypes[0]) if output_dtypes else None

    ndim = x.ndim
    dim_pos = activate_dim % ndim

    # Move activation axis to the last position for easy front/back or interleaved splitting.
    xf = np.moveaxis(x.astype(np.float32), dim_pos, -1)

    is_int_x = np.issubdtype(x.dtype, np.integer)
    if is_int_x:
        if weight_scale is not None:
            ws = (
                np.moveaxis(np.asarray(weight_scale).astype(np.float32), dim_pos, -1)
                if np.asarray(weight_scale).ndim == x.ndim
                else np.asarray(weight_scale).astype(np.float32)
            )
            xf = xf * ws
        if activation_scale is not None:
            act_s = np.asarray(activation_scale).astype(np.float32)
            xf = xf * act_s[..., None]
        if bias is not None:
            b = (
                np.moveaxis(np.asarray(bias).astype(np.float32), dim_pos, -1)
                if np.asarray(bias).ndim == x.ndim
                else np.asarray(bias).astype(np.float32)
            )
            xf = xf + b
    # else: fp16/bf16/fp32 x -- already dequantized, dequant params are ignored (FloatDequant).

    hidden2 = xf.shape[-1]
    H = hidden2 // 2

    if swiglu_mode == 1:
        x_glu = xf[..., 0::2]
        x_linear = xf[..., 1::2]
    else:
        if activate_left:
            x_glu = xf[..., :H]
            x_linear = xf[..., H:]
        else:
            x_glu = xf[..., H:]
            x_linear = xf[..., :H]

    if swiglu_mode == 0:
        y = x_glu * _sigmoid(x_glu) * x_linear
    elif swiglu_mode == 1 or swiglu_mode == 2:
        g = np.minimum(x_glu, clamp_limit)
        act = g * _sigmoid(glu_alpha * g)
        lin = np.clip(x_linear, -clamp_limit, clamp_limit) + glu_bias
        y = act * lin
    elif swiglu_mode == 3:
        silu = x_glu * _sigmoid(x_glu)
        act = np.minimum(silu, clamp_limit)
        lin = np.clip(x_linear, -clamp_limit, clamp_limit)
        y = act * lin
    else:
        raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")

    np_dtype = _target_np_dtype(dst_type, y_dtype_name)

    if quant_mode == "dynamic":
        row_max = np.max(np.abs(y), axis=-1, keepdims=True)
        scalar_max = SCALAR_MAX_NUM.get(dst_type, 127.0)
        scale = np.where(row_max > 0, row_max / scalar_max, 1.0).astype(np.float32)
        y_scaled = y / scale
        y_q = _cast_quant(y_scaled, np_dtype)
        scale_out = scale[..., 0].astype(np.float32)
    else:
        qs = (
            np.asarray(quant_scale).astype(np.float32)
            if quant_scale is not None
            else np.float32(1.0)
        )
        qo = (
            np.asarray(quant_offset).astype(np.float32)
            if quant_offset is not None
            else np.float32(0.0)
        )
        y_scaled = y / qs + qo
        y_q = _cast_quant(y_scaled, np_dtype)
        scale_out = np.zeros(y.shape[:-1], dtype=np.float32)

    # Move activation axis back to its original position.
    y_q = np.moveaxis(y_q, -1, dim_pos)

    return [y_q, scale_out]


__golden__ = {"kernel": {"dequant_swiglu_quant": "__golden_dequant_swiglu_quant"}}
