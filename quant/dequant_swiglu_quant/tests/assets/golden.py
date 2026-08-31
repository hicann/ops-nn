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

import numpy as np

__golden__ = {"kernel": {"dequant_swiglu_quant": "dequant_swiglu_quant_golden"}}

# dst_type -> max value used by dynamic quantization scale
SCALAR_MAX_NUM = {
    2: 127.0,  # int8
    34: 32768.0,  # hifloat8
    35: 57344.0,  # float8_e5m2
    36: 448.0,  # float8_e4m3fn
    40: 6.0,  # float4_e2m1
    41: 1.75,  # float4_e1m2
}

ROUND_MODE_MAP = {
    "rint": np.rint,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
}


def _sigmoid_div(x, alpha):
    # x / (1 + exp(-alpha * x)) == x * sigmoid(alpha * x), keeps the Div form used by the kernel
    with np.errstate(over="ignore", invalid="ignore"):
        return x / (1.0 + np.exp(-alpha * x))


def _target_np_dtype(dst_type, dtype_name):
    name = str(dtype_name) if dtype_name is not None else ""
    if dst_type == 2 or name == "int8":
        return np.int8
    if dst_type == 35 or name == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if dst_type == 36 or name == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if dst_type == 40 or name == "float4_e2m1":
        try:
            from en_dtypes import float4_e2m1

            return float4_e2m1
        except ImportError:
            from ml_dtypes import float4_e2m1fn

            return float4_e2m1fn
    if dst_type == 41 or name == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if dst_type == 34 or name == "hifloat8":
        from en_dtypes import hifloat8

        return hifloat8
    raise ValueError(
        f"Unsupported dst_type/output dtype: dst_type={dst_type}, dtype_name={name}"
    )


def _cast_quant(y_scaled, np_dtype, round_mode):
    if np_dtype == np.int8:
        round_fn = ROUND_MODE_MAP.get(round_mode, np.rint)
        return np.clip(round_fn(y_scaled), -128, 127).astype(np.int8)
    # fp8/fp4/hifloat8: numpy dtype casts round-to-nearest-even and saturate on overflow
    return y_scaled.astype(np.float32).astype(np_dtype)


def _swiglu_core(x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias):
    # swiglu_mode 0: y = SiLU(x_glu) * x_linear
    # swiglu_mode 1/2: x_glu clamp(max=L); x_linear clamp(-L,L)+gluBias;
    #                  y = x_glu * sigmoid(gluAlpha * x_glu) * (x_linear + gluBias)
    # swiglu_mode 3: gate = clamp(SiLU(x_glu), max=L); up = clamp(x_linear, -L, L); y = gate * up
    if swiglu_mode == 0:
        return _sigmoid_div(x_glu, 1.0) * x_linear
    if swiglu_mode == 1 or swiglu_mode == 2:
        g = np.minimum(x_glu, clamp_limit)
        lin = np.clip(x_linear, -clamp_limit, clamp_limit)
        return _sigmoid_div(g, glu_alpha) * (lin + glu_bias)
    if swiglu_mode == 3:
        gate = np.minimum(_sigmoid_div(x_glu, 1.0), clamp_limit)
        up = np.clip(x_linear, -clamp_limit, clamp_limit)
        return gate * up
    raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")


def _split_glu_linear(xf, swiglu_mode, activate_left):
    # split the activation axis (already the last axis) into glu/linear parts
    half = xf.shape[-1] // 2
    if swiglu_mode == 1:
        return xf[..., 0::2], xf[..., 1::2]
    if activate_left:
        return xf[..., :half], xf[..., half:]
    return xf[..., half:], xf[..., :half]


def _dequant(x_rows, bias_part, weight_scale_part, activation_scale_part):
    # int32 x: (+int bias in int32 domain) -> f32 -> *weight_scale -> *activation_scale -> (+float bias)
    # fp16/bf16 x: already dequantized, cast to f32 directly
    if np.issubdtype(x_rows.dtype, np.integer):
        xw = x_rows
        if bias_part is not None and np.issubdtype(
            np.asarray(bias_part).dtype, np.integer
        ):
            xw = xw + np.asarray(bias_part)
        xf = xw.astype(np.float32)
        if weight_scale_part is not None:
            xf = xf * weight_scale_part
        if activation_scale_part is not None:
            xf = xf * activation_scale_part
        if bias_part is not None and not np.issubdtype(
            np.asarray(bias_part).dtype, np.integer
        ):
            xf = xf + np.asarray(bias_part).astype(np.float32)
        return xf
    return x_rows.astype(np.float32)


def _group_param(param, group_idx, num_groups):
    # resolve a per-group parameter slice:
    #   2D (G, H)/(G, 1): take row group_idx; single-row 2D reuses row 0
    #   1D with length == num_groups: per-group scalar
    #   other 1D (H,) columns or (1,) scalar: broadcast as a whole
    if param is None:
        return None
    if param.ndim == 2:
        return param[group_idx] if param.shape[0] > 1 else param[0]
    if param.shape[0] == num_groups:
        return param[group_idx]
    return param


def _quant_last_axis(yv, is_dynamic, quant_scale_part, quant_offset_part, scalar_max):
    # returns (y_scaled, scale) where scale is None for static quantization
    if not is_dynamic:
        qs = quant_scale_part if quant_scale_part is not None else np.float32(1.0)
        qo = quant_offset_part if quant_offset_part is not None else np.float32(0.0)
        return yv / qs + qo, None
    if quant_scale_part is not None:
        yv = yv * quant_scale_part
    row_max = np.max(np.abs(yv), axis=-1)
    scale = np.where(row_max > 0, row_max / scalar_max, 1.0).astype(np.float32)
    return yv / scale[:, None], scale


def _run_last_axis(
    x,
    weight_scale,
    activation_scale,
    bias,
    quant_scale,
    quant_offset,
    group_index,
    swiglu_mode,
    activate_left,
    clamp_limit,
    glu_alpha,
    glu_bias,
    is_dynamic,
    scalar_max,
):
    raw_shape = x.shape
    x2 = x.reshape(-1, raw_shape[-1])
    rows, width = x2.shape
    half = width // 2

    if weight_scale is not None and weight_scale.ndim == 1:
        weight_scale = weight_scale.reshape(1, -1)
    if bias is not None and bias.ndim == 1:
        bias = bias.reshape(1, -1)
    if activation_scale is not None:
        activation_scale = activation_scale.reshape(-1, 1)

    groups = (
        group_index if group_index is not None else np.array([rows], dtype=np.int64)
    )
    num_groups = groups.shape[0]

    y = np.zeros((rows, half), dtype=np.float32)
    scale = np.zeros((rows,), dtype=np.float32)
    offset = 0
    for gi in range(num_groups):
        g = int(groups[gi])
        if g <= 0:
            continue
        x_part = x2[offset : offset + g]
        weight_scale_part = (
            weight_scale[gi if weight_scale.shape[0] > 1 else 0]
            if weight_scale is not None
            else None
        )
        activation_scale_part = (
            activation_scale[offset : offset + g]
            if activation_scale is not None
            else None
        )
        bias_part = bias[gi if bias.shape[0] > 1 else 0] if bias is not None else None

        xf = _dequant(x_part, bias_part, weight_scale_part, activation_scale_part)
        x_glu, x_linear = _split_glu_linear(xf, swiglu_mode, activate_left)
        yv = _swiglu_core(
            x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias
        )

        quant_scale_part = _group_param(quant_scale, gi, num_groups)
        quant_offset_part = _group_param(quant_offset, gi, num_groups)
        y_scaled, group_scale = _quant_last_axis(
            yv, is_dynamic, quant_scale_part, quant_offset_part, scalar_max
        )

        y[offset : offset + g] = y_scaled
        if group_scale is not None:
            scale[offset : offset + g] = group_scale
        offset += g

    y = y.reshape(raw_shape[:-1] + (half,))
    scale = scale.reshape(raw_shape[:-1])
    return y, scale


def _run_not_last_axis(
    x,
    weight_scale,
    activation_scale,
    bias,
    quant_scale,
    quant_offset,
    activate_dim,
    swiglu_mode,
    activate_left,
    clamp_limit,
    glu_alpha,
    glu_bias,
    is_dynamic,
    scalar_max,
):
    raw_shape = x.shape
    x2 = x.reshape(-1, raw_shape[-1])

    if weight_scale is not None and weight_scale.ndim == 1:
        weight_scale = weight_scale.reshape(1, -1)
    if bias is not None and bias.ndim == 1:
        bias = bias.reshape(1, -1)
    if activation_scale is not None:
        activation_scale = activation_scale.reshape(-1, 1)
    if quant_scale is not None and quant_scale.ndim == 1:
        quant_scale = quant_scale.reshape(1, -1)
    if quant_offset is not None and quant_offset.ndim == 1:
        quant_offset = quant_offset.reshape(1, -1)

    # dequant on the 2D view, then restore the raw shape and split along activate_dim
    xf = _dequant(x2, bias, weight_scale, activation_scale)
    xd = xf.reshape(raw_shape)

    halves = np.split(xd, 2, axis=activate_dim)
    if activate_left:
        x_glu, x_linear = halves[0], halves[1]
    else:
        x_glu, x_linear = halves[1], halves[0]
    y = _swiglu_core(x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias)

    if not is_dynamic:
        qs = quant_scale if quant_scale is not None else np.float32(1.0)
        qo = quant_offset if quant_offset is not None else np.float32(0.0)
        y = y / qs + qo
        scale = np.zeros(y.shape[:-1], dtype=np.float32)
    else:
        if quant_scale is not None:
            y = y * quant_scale
        row_max = np.max(np.abs(y), axis=-1, keepdims=True)
        scale = np.where(row_max > 0, row_max / scalar_max, 1.0).astype(np.float32)
        y = y / scale
        scale = scale[..., 0]
    return y, scale


def dequant_swiglu_quant_golden(
    x,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    *,
    activate_left=False,
    quant_mode="static",
    dst_type=2,
    round_mode="rint",
    activate_dim=-1,
    swiglu_mode=0,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
    **kwargs,
):
    """
    Golden function for dequant_swiglu_quant.
    All the parameters (names and order) follow @dequant_swiglu_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor list [y, scale]
    """
    x = np.asarray(x)
    ndim = x.ndim
    if activate_dim < 0:
        activate_dim += ndim

    weight_scale = (
        None if weight_scale is None else np.asarray(weight_scale).astype(np.float32)
    )
    activation_scale = (
        None
        if activation_scale is None
        else np.asarray(activation_scale).astype(np.float32)
    )
    bias = None if bias is None else np.asarray(bias)
    quant_scale = (
        None if quant_scale is None else np.asarray(quant_scale).astype(np.float32)
    )
    quant_offset = (
        None if quant_offset is None else np.asarray(quant_offset).astype(np.float32)
    )
    group_index = (
        None
        if group_index is None
        else np.asarray(group_index).reshape(-1).astype(np.int64)
    )

    is_dynamic = str(quant_mode).strip().lower() in ("dynamic", "1")

    output_dtypes = kwargs.get("output_dtypes")
    y_dtype_name = str(output_dtypes[0]) if output_dtypes else None
    np_dtype = _target_np_dtype(dst_type, y_dtype_name)
    scalar_max = SCALAR_MAX_NUM.get(dst_type, 127.0)

    if activate_dim == ndim - 1:
        y, scale = _run_last_axis(
            x,
            weight_scale,
            activation_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            swiglu_mode,
            activate_left,
            clamp_limit,
            glu_alpha,
            glu_bias,
            is_dynamic,
            scalar_max,
        )
    else:
        y, scale = _run_not_last_axis(
            x,
            weight_scale,
            activation_scale,
            bias,
            quant_scale,
            quant_offset,
            activate_dim,
            swiglu_mode,
            activate_left,
            clamp_limit,
            glu_alpha,
            glu_bias,
            is_dynamic,
            scalar_max,
        )

    y = _cast_quant(y, np_dtype, round_mode)
    return [y, scale]
