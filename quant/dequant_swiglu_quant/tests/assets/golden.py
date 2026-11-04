#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
TTK golden / TestSpec for dequant_swiglu_quant — Kernel / GEIR / ACLNN(V1+V2) / E2E.

Math core (torch, CPU fp32):
    dequant   : int32 x + int bias (int32 domain) -> f32 -> *weightScale -> *activationScale
                -> + float bias;  fp16/bf16 x: direct cast to f32 (already dequantized).
    swiglu    : mode0  y = SiLU(gate) * linear            (gate per activate_left)
                mode1  odd/even split, clamp + sigmoid(gluAlpha) + gluBias
                mode2  half split,   clamp + sigmoid(gluAlpha) + gluBias
                mode3  half split,   clamp(SiLU(gate)) * clamp(linear)
    quant     : static  y / quantScale + quantOffset
                dynamic scale = rowmax(|y|) / scalarMax(dstType) (zero-row kept at 1.0),
                        y / scale
Saturation cast (device contract, op_kernel/arch35/dequant_swiglu_quant_common.h):
    INT8     : round_mode -> clip [-128, 127]
    FP8/HIF8 : clip to +/-max_norm then cast (SatMode::SAT; e4m3fn 448 / e5m2 57344 /
               hifloat8 32768). NO NaN on overflow: verified on Ascend950PR that
               fp16 [[1000,1000]] static scale=1 dstType=36 yields 448 (not NaN).
    FP4      : f32 -> bfloat16 (CAST_RINT) -> fp4 grid with round_mode
               (device two-step cast: CAST_FP32_TO_BF16 then CAST_BF16_TO_FP4_<mode>).
    round_mode enum: 0/rint=nearest-even, 1/round=half-away-from-zero,
                     2=floor, 3=ceil, 4=trunc (verified on device).

E2E device-side output representations (torch_npu, verified on Ascend950PR):
    INT8 -> torch.int8; FP8 e4m3fn/e5m2 -> torch native fp8 dtypes;
    FP4  -> packed uint8 (even elem low nibble, odd elem high nibble, last dim halved);
    HIFLOAT8 -> uint8 bit-view.

scaleOut contract: dynamic mode must be compared (fp32); static mode does not
compute scaleOut (per aclnnDequantSwigluQuantV2 doc) -> compare skips it.
"""

import numpy as np

__golden__ = {
    "kernel": {"dequant_swiglu_quant": "dequant_swiglu_quant_golden"},
    "aclnn": {
        "aclnnDequantSwigluQuant": "aclnn_dequant_swiglu_quant_golden",
        "aclnnDequantSwigluQuantV2": "aclnn_dequant_swiglu_quant_v2_golden",
    },
    "e2e": {"torch_npu.npu_dequant_swiglu_quant": "npu_dequant_swiglu_quant_golden"},
}

__input__ = {
    "aclnn": {
        "aclnnDequantSwigluQuant": "dequant_swiglu_quant_group_index_input",
        "aclnnDequantSwigluQuantV2": "dequant_swiglu_quant_group_index_input",
    },
    "e2e": {
        "torch_npu.npu_dequant_swiglu_quant": "dequant_swiglu_quant_group_index_input"
    },
}

__spec__ = {
    "dequant_swiglu_quant": "DequantSwigluQuantKernelSpec",
    "aclnnDequantSwigluQuant": "AclnnDequantSwigluQuantSpec",
    "aclnnDequantSwigluQuantV2": "AclnnDequantSwigluQuantV2Spec",
    "torch_npu.npu_dequant_swiglu_quant": "DequantSwigluQuantE2ESpec",
}

# dst_type -> max value used by dynamic quantization scale (kernel scalarMaxNum_)
SCALAR_MAX_NUM = {
    2: 127.0,  # int8
    34: 32768.0,  # hifloat8
    35: 57344.0,  # float8_e5m2
    36: 448.0,  # float8_e4m3fn
    40: 6.0,  # float4_e2m1
    41: 1.75,  # float4_e1m2
}

# target finite range used by the saturation cast (SatMode::SAT on device)
SAT_RANGE = {
    2: (-128.0, 127.0),  # int8
    34: (-32768.0, 32768.0),  # hifloat8
    35: (-57344.0, 57344.0),  # float8_e5m2
    36: (-448.0, 448.0),  # float8_e4m3fn
    40: (-6.0, 6.0),  # float4_e2m1
    41: (-1.75, 1.75),  # float4_e1m2
}

ROUND_MODE_MAP = {
    "rint": np.rint,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
}

# E2E round_mode arrives as the op-plugin int enum (verified on device)
E2E_ROUND_MODE_ENUM = {0: "rint", 1: "round", 2: "floor", 3: "ceil", 4: "trunc"}

# E2E dst_type arrives as a torch ScalarType enum (verified on device);
# values map to the CANN dst_type codes used by the kernel/ACLNN layer.
TORCH_DTYPE_ENUM_TO_DST_TYPE = {
    None: 2,  # default INT8
    1: 2,  # int8
    23: 35,  # float8_e5m2
    24: 36,  # float8_e4m3fn
    290: 34,  # hifloat8
    291: 35,  # float8_e5m2 (npu alias)
    292: 36,  # float8_e4m3fn (npu alias)
    296: 40,  # float4_e2m1 (npu packed)
    297: 41,  # float4_e1m2 (npu packed)
}

DATA_TYPE_INT_TO_STR = {
    2: "int8",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}


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


# ---------------------------------------------------------------------------
# FP4 / HIFLOAT8 grid helpers (round modes other than RNE on the target grid)
# ---------------------------------------------------------------------------
def _lowbit_grid(dst_type):
    """Non-negative representable values of the target low-bit dtype, ascending.

    The full 256-code enumeration contains both -0.0 and +0.0 plus mirrored
    negatives; directed rounding is done on the magnitude grid with the sign
    restored afterwards, so only the non-negative half (with +0.0) is needed.
    """
    cache = getattr(_lowbit_grid, "_cache", None)
    if cache is None:
        cache = {}
        _lowbit_grid._cache = cache
    if dst_type in cache:
        return cache[dst_type]
    np_dtype = _target_np_dtype(dst_type, None)
    raw = np.arange(256, dtype=np.uint8)
    vals = raw.view(np_dtype).astype(np.float32)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    # prepend an explicit +0.0 (np.unique may keep -0.0 which encodes 0x8)
    vals = np.concatenate([np.zeros(1, dtype=np.float32), np.unique(vals)])
    cache[dst_type] = vals
    return vals


def _grid_quantize(y, dst_type, round_mode):
    """Quantize fp32 values onto the low-bit grid with the given round mode.

    rint -> nearest-even (native ml_dtypes/en_dtypes cast);
    round -> nearest, ties away from zero; floor/ceil/trunc -> directed
    rounding onto representable values (clipped to the finite range).
    """
    lo, hi = SAT_RANGE[dst_type]
    if round_mode in ("rint", "even", None):
        return np.clip(y, lo, hi).astype(_target_np_dtype(dst_type, None))
    grid = _lowbit_grid(dst_type)
    # Directed / half-away rounding: candidates on the MAGNITUDE grid; the sign
    # is restored afterwards. Value-domain floor/ceil for negatives maps to the
    # opposite magnitude candidate (floor(-v) == -ceil_mag(v)).
    # signbit keeps the sign of -0.0 (device keeps -0 in low-bit encodings).
    sign = np.where(np.signbit(y), -1.0, 1.0)
    a = np.abs(y)
    a = np.clip(a, 0.0, hi)
    n = grid.size
    # magnitude-floor: largest grid point <= |v|
    idx_f = np.clip(np.searchsorted(grid, a, side="right") - 1, 0, n - 1)
    # magnitude-ceil: smallest grid point >= |v|
    idx_c = np.clip(np.searchsorted(grid, a, side="left"), 0, n - 1)
    g_f = grid[idx_f]
    g_c = grid[idx_c]
    if round_mode in ("round", "nearest"):
        # ties away from zero on the magnitude grid
        mid = (g_f + g_c) / 2.0
        take_hi = (a > mid) | ((a == mid) & (g_c >= g_f))
        out_mag = np.where(take_hi, g_c, g_f)
    elif round_mode == "floor":
        out_mag = np.where(sign < 0, g_c, g_f)
    elif round_mode == "ceil":
        out_mag = np.where(sign < 0, g_f, g_c)
    elif round_mode == "trunc":
        out_mag = g_f
    else:
        raise ValueError(f"Unsupported round_mode: {round_mode}")
    out = out_mag * sign
    return out.astype(_target_np_dtype(dst_type, None))


def _cast_quant(y_scaled, dst_type, round_mode):
    """Saturating cast to the destination quant type (device SatMode::SAT)."""
    round_mode = "rint" if round_mode is None else str(round_mode)
    # device saturating casts map NaN (0/0 from all-zero dynamic rows) to 0
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, copy=False)
    lo, hi = SAT_RANGE[dst_type]
    if dst_type == 2:
        round_fn = ROUND_MODE_MAP.get(round_mode, np.rint)
        return np.clip(round_fn(y_scaled), lo, hi).astype(np.int8)
    if dst_type in (35, 36):
        # FP8: SAT + CAST_RINT; clip first so ml_dtypes never emits NaN/Inf
        return (
            np.clip(y_scaled, lo, hi)
            .astype(np.float32)
            .astype(_target_np_dtype(dst_type, None))
        )
    if dst_type == 34:
        # HIFLOAT8: device uses CAST_ROUND (half away from zero), doc: only "round"
        mode = round_mode if round_mode in ("round",) else "round"
        return _grid_quantize(y_scaled, dst_type, mode)
    if dst_type in (40, 41):
        # FP4: device two-step cast f32 -> bfloat16(CAST_RINT) -> fp4(round_mode)
        from ml_dtypes import bfloat16

        y_bf16 = y_scaled.astype(np.float32).astype(bfloat16).astype(np.float32)
        return _grid_quantize(y_bf16, dst_type, round_mode)
    raise ValueError(f"Unsupported dst_type: {dst_type}")


# ---------------------------------------------------------------------------
# Torch math core (device-verified on Ascend950PR, bit-exact for fp32 path)
# ---------------------------------------------------------------------------
def _torch_sigmoid_div(x, alpha):
    # x / (1 + exp(-alpha * x)) == x * sigmoid(alpha * x), keeps the Div form
    import torch

    return x / (1.0 + torch.exp(-alpha * x))


def _torch_swiglu_core(x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias):
    import torch

    if swiglu_mode == 0:
        return _torch_sigmoid_div(x_glu, 1.0) * x_linear
    if swiglu_mode == 1 or swiglu_mode == 2:
        g = torch.clamp(x_glu, max=clamp_limit)
        lin = torch.clamp(x_linear, min=-clamp_limit, max=clamp_limit)
        return _torch_sigmoid_div(g, glu_alpha) * (lin + glu_bias)
    if swiglu_mode == 3:
        gate = torch.clamp(_torch_sigmoid_div(x_glu, 1.0), max=clamp_limit)
        up = torch.clamp(x_linear, min=-clamp_limit, max=clamp_limit)
        return gate * up
    raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")


def _torch_split_glu_linear(xf, swiglu_mode, activate_left):
    """Split the activation axis (already moved to the last position)."""
    if swiglu_mode == 1:
        return xf[..., 0::2], xf[..., 1::2]
    half = xf.shape[-1] // 2
    if activate_left:
        return xf[..., :half], xf[..., half:]
    return xf[..., half:], xf[..., :half]


def _torch_dequant(x_rows, bias_part, weight_scale_part, activation_scale_part):
    """int32 x: (+int bias in int32 domain) -> f32 -> *ws -> *as -> (+float bias)."""
    import torch

    if x_rows.dtype == torch.int32 or x_rows.dtype == torch.int64:
        xw = x_rows
        if bias_part is not None and torch.is_floating_point(bias_part) is False:
            xw = xw + bias_part
        xf = xw.to(torch.float32)
        if weight_scale_part is not None:
            xf = xf * weight_scale_part
        if activation_scale_part is not None:
            xf = xf * activation_scale_part
        if bias_part is not None and torch.is_floating_point(bias_part):
            xf = xf + bias_part.to(torch.float32)
        return xf
    return x_rows.to(torch.float32)


def _group_param(param, group_idx, num_groups):
    # 2D (G, H)/(G, 1): take row group_idx (single-row 2D reuses row 0);
    # 1D with length == num_groups: per-group scalar; other 1D: broadcast whole.
    if param is None:
        return None
    if param.ndim == 2:
        return param[group_idx] if param.shape[0] > 1 else param[0]
    if param.shape[0] == num_groups:
        return param[group_idx]
    return param


def _torch_quant_last_axis(
    yv, is_dynamic, quant_scale_part, quant_offset_part, scalar_max
):
    """Per-row quantization along the last axis. Returns (y_scaled, scale)."""
    import torch

    if not is_dynamic:
        # device divides by quantScale (op_kernel/arch35/dequant_swiglu_quant_common.h
        # StaticQuant* use Reg::Div; the comments there read "*" but the
        # instruction is a divide, confirmed byte-exact on Ascend950PR)
        qs = quant_scale_part if quant_scale_part is not None else 1.0
        y = yv / qs
        if quant_offset_part is not None:
            # the kernel emits the offset add only when the tensor exists; adding
            # 0.0 unconditionally would turn the device's -0.0 into +0.0 and break
            # the bit-exact compare for signed low-bit types (fp8/fp4)
            y = y + quant_offset_part
        return y, None
    if quant_scale_part is not None:
        yv = yv * quant_scale_part
    row_max = torch.max(torch.abs(yv), dim=-1, keepdim=True).values
    # device contract (kernel Reduce-MAX over an abs buffer seeded with 0,
    # then Div by scalarMax; verified on Ascend950PR): an all-zero row yields
    # scale = 0 and y = 0/0 = NaN, which the saturating cast maps to 0.
    scale = (row_max / scalar_max).to(torch.float32)
    return yv / scale, scale


def _torch_run(
    x_t,
    weight_scale_t,
    activation_scale_t,
    bias_t,
    quant_scale_t,
    quant_offset_t,
    group_index,
    swiglu_mode,
    activate_left,
    clamp_limit,
    glu_alpha,
    glu_bias,
    is_dynamic,
    scalar_max,
    activate_dim,
):
    """Full dequant->swiglu->quant chain in torch fp32.

    Returns (y_scaled torch fp32, scale torch fp32 or None for static).
    """
    import torch

    ndim = x_t.dim()
    if activate_dim < 0:
        activate_dim += ndim
    last_axis = activate_dim == ndim - 1

    raw_shape = tuple(x_t.shape)
    x2 = x_t.reshape(-1, raw_shape[-1])
    rows, width = x2.shape

    # dequant params are laid out on the last (memory) axis
    if weight_scale_t is not None and weight_scale_t.dim() == 1:
        weight_scale_t = weight_scale_t.reshape(1, -1)
    if bias_t is not None and bias_t.dim() == 1:
        bias_t = bias_t.reshape(1, -1)
    if activation_scale_t is not None:
        activation_scale_t = activation_scale_t.reshape(-1, 1)
    if not last_axis:
        if quant_scale_t is not None and quant_scale_t.dim() == 1:
            quant_scale_t = quant_scale_t.reshape(1, -1)
        if quant_offset_t is not None and quant_offset_t.dim() == 1:
            quant_offset_t = quant_offset_t.reshape(1, -1)

    if group_index is not None and last_axis:
        # grouped last-axis path: parameters sliced per group along rows
        groups = group_index.reshape(-1).to(torch.int64)
        num_groups = groups.numel()
        y_rows = []
        scale_rows = []
        offset = 0
        for gi in range(num_groups):
            g = int(groups[gi])
            if g <= 0:
                continue
            x_part = x2[offset : offset + g]
            ws_part = (
                weight_scale_t[gi if weight_scale_t.shape[0] > 1 else 0]
                if weight_scale_t is not None
                else None
            )
            as_part = (
                activation_scale_t[offset : offset + g]
                if activation_scale_t is not None
                else None
            )
            b_part = (
                bias_t[gi if bias_t.shape[0] > 1 else 0] if bias_t is not None else None
            )
            xf = _torch_dequant(x_part, b_part, ws_part, as_part)
            x_glu, x_linear = _torch_split_glu_linear(xf, swiglu_mode, activate_left)
            yv = _torch_swiglu_core(
                x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias
            )
            qs_part = _group_param(quant_scale_t, gi, num_groups)
            qo_part = _group_param(quant_offset_t, gi, num_groups)
            y_scaled, group_scale = _torch_quant_last_axis(
                yv, is_dynamic, qs_part, qo_part, scalar_max
            )
            y_rows.append(y_scaled)
            if group_scale is not None:
                scale_rows.append(group_scale[:, 0])
            offset += g
        y = torch.cat(y_rows, dim=0)
        scale = torch.cat(scale_rows, dim=0) if scale_rows else None
        # same N-D restore as the plain last-axis path: the device emits
        # y (X1..Xn-1, W/2) and scale (X1..Xn-1), not the 2D row view
        grouped_shape = raw_shape[:-1] + (y.shape[-1],)
        if y.numel() == int(np.prod(grouped_shape)):
            y = y.reshape(grouped_shape)
            scale = scale.reshape(grouped_shape[:-1]) if scale is not None else None
        return y, scale

    # non-grouped / non-last-axis path
    xf = _torch_dequant(x2, bias_t, weight_scale_t, activation_scale_t)
    if last_axis:
        x_glu, x_linear = _torch_split_glu_linear(xf, swiglu_mode, activate_left)
        # y/scale must carry the input's leading axes: the device emits
        # y (X1..Xn-1, W/2) and scale (X1..Xn-1), not the flattened row view
        # used for the computation (2D inputs made the flattening invisible).
        out_raw_shape = raw_shape[:-1] + (x_glu.shape[-1],)
    else:
        # restore raw shape and split along activate_dim; compute on a 2D row
        # view (each row = one last-axis line), then restore the raw shape
        xd = xf.reshape(raw_shape)
        halves = torch.chunk(xd, 2, dim=activate_dim)
        if activate_left:
            x_glu, x_linear = halves[0], halves[1]
        else:
            x_glu, x_linear = halves[1], halves[0]
        out_raw_shape = tuple(x_glu.shape)
        x_glu = x_glu.reshape(-1, x_glu.shape[-1])
        x_linear = x_linear.reshape(-1, x_linear.shape[-1])
    yv = _torch_swiglu_core(
        x_glu, x_linear, swiglu_mode, clamp_limit, glu_alpha, glu_bias
    )

    if not is_dynamic:
        # static quant divides by quantScale, and only adds the offset when the
        # tensor exists (see _torch_quant_last_axis for the signed-zero reason)
        qs = quant_scale_t if quant_scale_t is not None else 1.0
        y = yv / qs
        if quant_offset_t is not None:
            y = y + quant_offset_t
        scale = None
    else:
        if quant_scale_t is not None:
            yv = yv * quant_scale_t
        row_max = torch.max(torch.abs(yv), dim=-1, keepdim=True).values
        # device contract: all-zero row -> scale = 0 (no guard), y = 0/0 = NaN
        # -> saturating cast maps NaN to 0 (verified on Ascend950PR)
        scale = (row_max / scalar_max).to(torch.float32)
        y = yv / scale
        scale = scale[:, 0]
    if out_raw_shape is not None:
        # y -> (X1,..,Xad/2,..,Xn); scale -> its leading axes (verified on
        # Ascend950PR: act_dim=-2 on (4,64,128) yields scale (4,32))
        y = y.reshape(out_raw_shape)
        scale = scale.reshape(out_raw_shape[:-1]) if scale is not None else None
    return y, scale


def _normalize_attrs(
    quant_mode,
    dst_type,
    round_mode,
    activate_dim,
    swiglu_mode=0,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
):
    """Normalize the attribute variants accepted across Kernel/ACLNN/E2E paths."""
    if isinstance(quant_mode, str):
        is_dynamic = quant_mode.strip().lower() in ("dynamic", "1")
    else:
        is_dynamic = int(quant_mode) in (1,)
    dst_type = int(dst_type) if dst_type is not None else 2
    if isinstance(round_mode, int):
        round_mode = E2E_ROUND_MODE_ENUM.get(round_mode, "rint")
    round_mode = "rint" if round_mode is None else str(round_mode)
    return (
        is_dynamic,
        dst_type,
        round_mode,
        int(activate_dim) if activate_dim is not None else -1,
        int(swiglu_mode),
        float(clamp_limit),
        float(glu_alpha),
        float(glu_bias),
    )


def _core_compute(x_t, params_t, attrs):
    """Shared compute: torch tensors in -> (y numpy dst dtype, scale numpy fp32).

    attrs: (is_dynamic, dst_type, round_mode, activate_dim, swiglu_mode,
            clamp_limit, glu_alpha, glu_bias, activate_left)
    """
    (
        is_dynamic,
        dst_type,
        round_mode,
        activate_dim,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
        activate_left,
    ) = attrs
    scalar_max = SCALAR_MAX_NUM.get(dst_type, 127.0)
    x_t, ws_t, as_t, b_t, qs_t, qo_t, gi_t = params_t
    y_scaled, scale = _torch_run(
        x_t,
        ws_t,
        as_t,
        b_t,
        qs_t,
        qo_t,
        gi_t,
        swiglu_mode,
        activate_left,
        clamp_limit,
        glu_alpha,
        glu_bias,
        is_dynamic,
        scalar_max,
        activate_dim,
    )
    y_np = y_scaled.cpu().numpy()
    y_q = _cast_quant(y_np, dst_type, round_mode)
    scale_np = (
        scale.cpu().numpy().astype(np.float32)
        if scale is not None
        else np.zeros(y_np.shape[:-1], dtype=np.float32)
    )
    return y_q, scale_np


def _torch_params_from(
    x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index
):
    """Convert numpy/torch inputs to torch fp32/int tensors (shared by all paths)."""
    import torch

    def to_t(v, dtype=None, int_ok=False):
        if v is None:
            return None
        if hasattr(v, "detach"):  # torch.Tensor (any device)
            t = v.detach().cpu()
            if int_ok and not torch.is_floating_point(t):
                return t
            if dtype is not None:
                return t.to(dtype)
            return t.to(torch.float32)
        arr = np.asarray(v)
        if int_ok and np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))

    x_t = to_t(x, int_ok=True)  # int32 stays int for the dequant path
    if x_t is not None and torch.is_floating_point(x_t):
        x_t = x_t.to(torch.float32)
    ws_t = to_t(weight_scale)
    as_t = to_t(activation_scale)
    b_t = to_t(bias, int_ok=True)
    if b_t is not None and torch.is_floating_point(b_t):
        b_t = b_t.to(torch.float32)
    qs_t = to_t(quant_scale)
    qo_t = to_t(quant_offset)
    gi_t = to_t(group_index, int_ok=True)
    return x_t, ws_t, as_t, b_t, qs_t, qo_t, gi_t


# ---------------------------------------------------------------------------
# Kernel / GEIR golden (numpy in, numpy out)
# ---------------------------------------------------------------------------
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
    Golden function for dequant_swiglu_quant (kernel/GEIR path, numpy I/O).
    All the parameters (names and order) follow @dequant_swiglu_quant_def.cpp
    without outputs.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor list [y, scale]
    """
    attrs = _normalize_attrs(
        quant_mode,
        dst_type,
        round_mode,
        activate_dim,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
    ) + (activate_left,)
    output_dtypes = kwargs.get("output_dtypes")
    y_dtype_name = str(output_dtypes[0]) if output_dtypes else None
    # resolve dst_type from the output dtype name (authoritative for kernel CSV)
    if y_dtype_name and y_dtype_name != "None":
        for code, name in DATA_TYPE_INT_TO_STR.items():
            if name == y_dtype_name:
                dst_type = code
                break
        attrs = (
            attrs[0],
            dst_type,
            attrs[2],
            attrs[3],
            attrs[4],
            attrs[5],
            attrs[6],
            attrs[7],
            attrs[8],
        )
    params_t = _torch_params_from(
        x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index
    )
    y_q, scale_np = _core_compute(params_t[0], params_t, attrs)
    return [y_q, scale_np]


# ---------------------------------------------------------------------------
# ACLNN goldens (torch.Tensor or numpy in — C header parameter order)
# ---------------------------------------------------------------------------
def aclnn_dequant_swiglu_quant_golden(
    x,
    weightScaleOptional=None,
    activationScaleOptional=None,
    biasOptional=None,
    quantScaleOptional=None,
    quantOffsetOptional=None,
    groupIndexOptional=None,
    activateLeft=False,
    quantModeOptional="static",
    yOut=None,
    scaleOut=None,
    **kwargs,
):
    """Golden for aclnnDequantSwigluQuant (V1: dstType fixed to INT8).

    Positional args follow aclnnDequantSwigluQuantGetWorkspaceSize without
    workspaceSize/executor. Tensors arrive as torch.Tensor (torch-native
    dtypes) or numpy.ndarray (non-native dtypes).
    """
    attrs = _normalize_attrs(
        quantModeOptional if quantModeOptional is not None else "static",
        2,
        "rint",
        -1,
        0,
        7.0,
        1.702,
        1.0,
    ) + (bool(activateLeft),)
    params_t = _torch_params_from(
        x,
        weightScaleOptional,
        activationScaleOptional,
        biasOptional,
        quantScaleOptional,
        quantOffsetOptional,
        groupIndexOptional,
    )
    y_q, scale_np = _core_compute(params_t[0], params_t, attrs)
    return [y_q, scale_np]


def aclnn_dequant_swiglu_quant_v2_golden(
    x,
    weightScaleOptional=None,
    activationScaleOptional=None,
    biasOptional=None,
    quantScaleOptional=None,
    quantOffsetOptional=None,
    groupIndexOptional=None,
    activateLeft=False,
    quantModeOptional="static",
    dstType=2,
    roundModeOptional="rint",
    activateDim=-1,
    swigluMode=0,
    clampLimit=7.0,
    gluAlpha=1.702,
    gluBias=1.0,
    yOut=None,
    scaleOut=None,
    **kwargs,
):
    """Golden for aclnnDequantSwigluQuantV2.

    Positional args follow aclnnDequantSwigluQuantV2GetWorkspaceSize without
    workspaceSize/executor. dstType in {2,34,35,36,40,41}; quantMode None ->
    static; roundMode None -> rint.
    """
    attrs = _normalize_attrs(
        quantModeOptional if quantModeOptional is not None else "static",
        dstType,
        roundModeOptional,
        activateDim,
        swigluMode,
        clampLimit,
        gluAlpha,
        gluBias,
    ) + (bool(activateLeft),)
    params_t = _torch_params_from(
        x,
        weightScaleOptional,
        activationScaleOptional,
        biasOptional,
        quantScaleOptional,
        quantOffsetOptional,
        groupIndexOptional,
    )
    y_q, scale_np = _core_compute(params_t[0], params_t, attrs)
    return [y_q, scale_np]


# ---------------------------------------------------------------------------
# E2E golden (torch_npu python signature; device-side output representations)
# ---------------------------------------------------------------------------
def _pack_fp4_to_uint8(ele):
    """Pack a native FP4 tensor into the op-plugin uint8 representation:
    even element -> low nibble, odd element -> high nibble (verified on device)."""
    if not isinstance(ele, np.ndarray) or "float4" not in str(ele.dtype):
        return ele
    if ele.shape[-1] % 2:
        raise ValueError("FP4 packing requires an even last dim")
    raw = ele.view(np.uint8) & 0x0F
    pairs = raw.reshape(raw.shape[:-1] + (-1, 2))
    packed = pairs[..., 0].astype(np.uint16) | (pairs[..., 1].astype(np.uint16) << 4)
    return packed.astype(np.uint8)


def npu_dequant_swiglu_quant_golden(
    x,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    *,
    activate_left=False,
    quant_mode=0,
    dst_type=None,
    round_mode=None,
    activate_dim=None,
    swiglu_mode=0,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
    **kwargs,
):
    """E2E golden for torch_npu.npu_dequant_swiglu_quant.

    Follows the op-plugin python signature (quant_mode int 0/1; dst_type torch
    ScalarType enum or None; round_mode int enum or None). Returns outputs in
    the device-side representation: int8 / native fp8 dtypes / packed uint8
    (FP4) / uint8 bit-view (HIFLOAT8), plus fp32 scale.
    """
    if isinstance(quant_mode, str):
        quant_mode = 1 if quant_mode.strip().lower() == "dynamic" else 0
    dst = TORCH_DTYPE_ENUM_TO_DST_TYPE.get(dst_type, 2)
    rm = (
        round_mode
        if round_mode is None
        else E2E_ROUND_MODE_ENUM.get(int(round_mode), "rint")
    )
    attrs = _normalize_attrs(
        quant_mode,
        dst,
        rm,
        activate_dim if activate_dim is not None else -1,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
    ) + (bool(activate_left),)
    params_t = _torch_params_from(
        x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index
    )
    y_q, scale_np = _core_compute(params_t[0], params_t, attrs)

    # match the device-side representation
    if dst in (40, 41):
        y_q = _pack_fp4_to_uint8(y_q)
    elif dst == 34:
        y_q = np.ascontiguousarray(y_q).view(np.uint8)
    return [np.ascontiguousarray(y_q), np.ascontiguousarray(scale_np)]


# ---------------------------------------------------------------------------
# Input normalization (group_index hygiene, shared by aclnn/e2e)
# ---------------------------------------------------------------------------
def _group_row_counts(values, rows):
    """Map generated values onto a valid per-group row partition.

    The kernel reads groupIndex[i] as the row count of group i and advances its
    row offset by it (op_kernel/arch35/dequant_swiglu_quant_static.h), and the
    API doc requires the element sum to stay within the row count: output rows
    beyond that sum are left as uninitialized memory. Randomly generated values
    break the constraint, so they are rescaled into a partition summing to
    exactly `rows`, which keeps every output element defined and comparable.
    Zero-count groups are kept (the kernel skips them) to cover that branch.
    """
    weights = np.abs(np.asarray(values).astype(np.int64).reshape(-1)) % 997
    if int(weights.sum()) == 0:
        weights = np.ones_like(weights)
    counts = (weights * int(rows)) // int(weights.sum())
    short = int(rows) - int(counts.sum())
    if short > 0:
        counts[np.argsort(-weights, kind="stable")[:short]] += 1
    return counts


def dequant_swiglu_quant_group_index_input(x, *extra, **kwargs):
    """Rewrite group_index in place as a valid row partition (see above).

    extra follows the C header order after x:
    (weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
    yOut, scaleOut, ...). Only the int-typed tensor is normalized; torch
    tensors are handled without np.asarray (bf16 tensors cannot convert).
    """
    gi = None
    for cand in extra[:6]:
        if cand is None or not hasattr(cand, "shape"):
            continue
        is_int = False
        if hasattr(cand, "dtype"):
            dstr = str(cand.dtype)
            is_int = dstr.startswith("torch.int") or dstr.startswith("int")
        if is_int:
            gi = cand
            break
    if gi is None:
        return
    xshape = tuple(x.shape) if hasattr(x, "shape") else np.asarray(x).shape
    dim_s = int(np.prod(xshape[:-1])) if len(xshape) > 1 else 1
    if hasattr(gi, "copy_"):
        import torch

        counts = _group_row_counts(gi.detach().cpu().numpy(), dim_s)
        gi.copy_(torch.as_tensor(counts, dtype=gi.dtype))
    else:
        gi[...] = _group_row_counts(gi, dim_s).astype(np.asarray(gi).dtype)


def dequant_swiglu_quant_kernel_input(*args, **kwargs):
    """Kernel/GEIR variant: same normalization, but the sequence is returned
    (the kernel path consumes the returned inputs, not the in-place write)."""
    dequant_swiglu_quant_group_index_input(*args, **kwargs)
    return list(args)


# ---------------------------------------------------------------------------
# Custom compare: per-output, per-dtype standards (see class docstrings)
# ---------------------------------------------------------------------------


def _ensure_np(t):
    """Safely convert torch tensors (incl. fp8/bf16/uint8 views) to numpy."""
    if not hasattr(t, "detach"):
        return np.asarray(t)
    import torch

    t = t.detach().cpu()
    s = str(t.dtype)
    if "float8_e4m3fn" in s:
        return t.view(torch.uint8).numpy().view(_np_f8e4m3fn())
    if "float8_e5m2" in s and "fnuz" not in s:
        return t.view(torch.uint8).numpy().view(_np_f8e5m2())
    if "e8m0" in s:
        # torch float8_e8m0fnu device tensors map to raw E8M0 exponent bytes
        return t.view(torch.uint8).numpy()
    if "bfloat16" in s:
        from ml_dtypes import bfloat16

        return t.view(torch.int16).numpy().view(bfloat16)
    if s == "torch.uint8":
        return t.numpy()
    return t.numpy()


def _np_f8e4m3fn():
    from ml_dtypes import float8_e4m3fn

    return float8_e4m3fn


def _np_f8e5m2():
    from ml_dtypes import float8_e5m2

    return float8_e5m2


def _raw_uint8(array):
    array = np.ascontiguousarray(np.asarray(array))
    return np.frombuffer(array.tobytes(), dtype=np.uint8)


def _bin_compare(actual, expected, output_name, valid_rows=None):
    a = _ensure_np(actual)
    e = _ensure_np(expected)
    if valid_rows is not None:
        a = a[valid_rows]
        e = e[valid_rows]
    ar, er = _raw_uint8(a), _raw_uint8(e)
    if ar.size != er.size:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{output_name} size mismatch: {ar.size} vs {er.size}",
        }
    diff = int(np.count_nonzero(ar != er))
    total = int(ar.size)
    precision = 100.0 if total == 0 else (total - diff) / total * 100.0
    return {
        "pass": diff == 0,
        "precision": precision,
        "error_info": None
        if diff == 0
        else f"{output_name} mismatch: {diff}/{total} bytes",
    }


def _quant_1lsb_compare(actual, expected, output_name):
    """FP8 quantized elements: |err| <= 1 LSB (requant semantics)."""

    a = _ensure_np(actual)
    e = _ensure_np(expected)
    if a.size != e.size:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{output_name} size mismatch: {a.size} vs {e.size}",
        }
    # 1 LSB of the fp8 grid is a step in the mantissa of the expected value
    af = a.astype(np.float32)
    ef = e.astype(np.float32)
    dtype_name = f"{a.dtype}{e.dtype}"
    # 1 LSB of the fp8 grid: relative step from the mantissa width
    # (e4m3fn: 3 mantissa bits -> 2^-3; e5m2: 2 bits -> 2^-2)
    lsb_rel = 2.0**-3 if "e4m3" in str(dtype_name) else 2.0**-2
    step = np.maximum(np.abs(ef) * lsb_rel, 2.0**-9)
    bad = np.abs(af - ef) > step
    precision = 100.0 * (1.0 - bad.mean()) if bad.size else 100.0
    return {
        "pass": not bad.any(),
        "precision": precision,
        "error_info": (
            None
            if not bad.any()
            else f"{output_name} >1LSB mismatch: {int(bad.sum())}/{bad.size}"
        ),
    }


def _scale_compare(actual, expected, output_name):
    a = _ensure_np(actual).astype(np.float32)
    e = _ensure_np(expected).astype(np.float32)
    if a.shape != e.shape:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{output_name} shape mismatch: {a.shape} vs {e.shape}",
        }
    same_inf = (np.isinf(a) & np.isinf(e)) | (~np.isinf(a) & ~np.isinf(e))
    rel = np.abs(a - e) / np.maximum(np.abs(e), 1e-30)
    bad = (rel > 1e-6) & same_inf
    precision = 100.0 * (1.0 - bad.mean()) if bad.size else 100.0
    return {
        "pass": not bad.any(),
        "precision": precision,
        "error_info": (
            None
            if not bad.any()
            else f"{output_name} mismatch: {int(bad.sum())}/{bad.size} rows"
        ),
    }


# ---------------------------------------------------------------------------
# Boundary-aware quant comparison
# ---------------------------------------------------------------------------
# The device computes the swish division and the dynamic-quant division with
# hardware instructions that are accurate to about 1 ULP but are not correctly
# rounded (Reg::Exp -> vexp; Reg::Div default mode -> vdiv + exponent rebuild
# without a sticky bit, see op_kernel/arch35/dequant_swiglu_quant_common.h and
# the CANN dav_3510 reg-compute implementation). A pre-cast value that sits on
# a rounding boundary can therefore legitimately land on either neighbouring
# code, while every value away from a boundary must match bit-exactly.
QUANT_BOUNDARY_ULPS = 4.0

# attribute spellings differ per path (kernel CSV / ACLNN C names / op-plugin)
_ATTR_ALIASES = {
    "quant_mode": ("quant_mode", "quantModeOptional", "quantMode"),
    "activate_left": ("activate_left", "activateLeft"),
    "round_mode": ("round_mode", "roundModeOptional", "roundMode"),
    "activate_dim": ("activate_dim", "activateDim"),
    "swiglu_mode": ("swiglu_mode", "swigluMode"),
    "clamp_limit": ("clamp_limit", "clampLimit"),
    "glu_alpha": ("glu_alpha", "gluAlpha"),
    "glu_bias": ("glu_bias", "gluBias"),
}


def _attr(attrs, name, default):
    for key in _ATTR_ALIASES.get(name, (name,)):
        if key in attrs and attrs[key] is not None:
            return attrs[key]
    return default


def _dst_type_of(golden_array):
    """dst_type code implied by the golden y dtype (None when ambiguous)."""
    name = str(_ensure_np(golden_array).dtype)
    for code, dtype_name in DATA_TYPE_INT_TO_STR.items():
        if dtype_name == name:
            return code
    return None


def _recompute_pre_cast(compare_context, dst_type):
    """Pre-cast fp32 y rebuilt from the compare context (None when unavailable).

    The first seven tensor slots are the op inputs in definition order on every
    path (kernel CSV, ACLNN V1/V2 C order, op-plugin signature), so the shared
    torch core can be replayed here.
    """
    if compare_context is None or dst_type is None:
        return None
    tensors = getattr(compare_context, "input_tensors", None)
    if not tensors:
        return None
    slots = list(tensors)[:7]
    if not slots or slots[0] is None or not hasattr(slots[0], "shape"):
        return None
    slots += [None] * (7 - len(slots))
    attrs = dict(getattr(compare_context, "attributes", None) or {})
    norm = _normalize_attrs(
        _attr(attrs, "quant_mode", "static"),
        dst_type,
        _attr(attrs, "round_mode", "rint"),
        _attr(attrs, "activate_dim", -1),
        _attr(attrs, "swiglu_mode", 0),
        _attr(attrs, "clamp_limit", 7.0),
        _attr(attrs, "glu_alpha", 1.702),
        _attr(attrs, "glu_bias", 1.0),
    )
    params_t = _torch_params_from(*slots)
    y_scaled, _ = _torch_run(
        *params_t,
        norm[4],
        bool(_attr(attrs, "activate_left", False)),
        norm[5],
        norm[6],
        norm[7],
        norm[0],
        SCALAR_MAX_NUM.get(dst_type, 127.0),
        norm[3],
    )
    return y_scaled.cpu().numpy().astype(np.float32)


def _quant_boundary_compare(actual, expected, pre, dst_type, round_mode, output_name):
    """Bit-exact except on rounding boundaries reachable within a few ULP.

    An element may differ from the golden code only if perturbing its pre-cast
    value by +/- QUANT_BOUNDARY_ULPS ULP reproduces exactly the device code.
    """
    a = _raw_uint8(_ensure_np(actual))
    e = _raw_uint8(_ensure_np(expected))
    if a.size != e.size:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{output_name} size mismatch: {a.size} vs {e.size}",
        }
    neq = a != e
    total = int(a.size)
    if not neq.any():
        return {"pass": True, "precision": 100.0, "error_info": None}
    eps = np.float32(2.0**-24 * QUANT_BOUNDARY_ULPS)
    tolerated = np.zeros_like(neq)
    for factor in (np.float32(1.0) - eps, np.float32(1.0) + eps):
        cand = _cast_quant(
            np.asarray(pre, dtype=np.float32) * factor, dst_type, round_mode
        )
        cand_raw = _raw_uint8(cand)
        if cand_raw.size != total:
            return {
                "pass": False,
                "precision": 0.0,
                "error_info": f"{output_name} boundary replay size mismatch: {cand_raw.size} vs {total}",
            }
        tolerated |= a == cand_raw
    bad = int(np.count_nonzero(neq & ~tolerated))
    boundary = int(np.count_nonzero(neq & tolerated))
    precision = (total - bad) / total * 100.0
    return {
        "pass": bad == 0,
        "precision": precision,
        "error_info": None
        if bad == 0
        else f"{output_name} mismatch: {bad}/{total} elements off the rounding boundary "
        f"({boundary} tolerated within {QUANT_BOUNDARY_ULPS:g} ULP)",
    }


def dequant_swiglu_quant_compare(
    y_npu, scale_npu, y_gold, scale_gold, compare_context=None, **kwargs
):
    """Per-output compare for dequant_swiglu_quant (all paths).

    Standards (documented per dtype, see Spec docstrings):
      - y int8        : binary (byte) equal
      - y fp8 e4m3/e5m2: requant, |err| <= 1 LSB
      - y fp4 packed  : nibble-exact (binary)
      - y hifloat8    : bit-view binary
      - scale (dynamic): fp32 compared (relative 1e-6)
      - scale (static) : NOT compared (contract: static does not compute scaleOut)
    Grouped cases compare only rows covered by group_index (rest is garbage
    memory per doc).
    """
    attrs = {}
    if compare_context is not None:
        attrs = dict(compare_context.attributes or {})
    quant_mode = attrs.get("quantModeOptional", attrs.get("quant_mode", "static"))
    if quant_mode is None:
        quant_mode = "static"
    is_dynamic = (
        quant_mode in ("dynamic", 1)
        if not isinstance(quant_mode, str)
        else quant_mode.strip().lower() == "dynamic"
    )

    results = []
    y_dtype = str(_ensure_np(y_gold).dtype)
    dst_type = _dst_type_of(y_gold)
    round_mode = _attr(attrs, "round_mode", "rint")
    if isinstance(round_mode, int):
        round_mode = E2E_ROUND_MODE_ENUM.get(round_mode, "rint")
    # boundary-aware standard needs the pre-cast value; it is only replayable
    # for one-byte-per-element dtypes (fp4 reaches compare nibble-packed)
    pre = None
    if dst_type is not None and dst_type != 40 and dst_type != 41:
        try:
            pre = _recompute_pre_cast(compare_context, dst_type)
        except Exception:  # replay is best-effort: fall back to the bit standard
            pre = None
    if pre is not None:
        results.append(
            _quant_boundary_compare(
                y_npu,
                y_gold,
                pre,
                dst_type,
                round_mode,
                f"y({DATA_TYPE_INT_TO_STR[dst_type]})",
            )
        )
    elif y_dtype.startswith("int8"):
        results.append(_bin_compare(y_npu, y_gold, "y"))
    elif "float4" in y_dtype:
        # golden already packed; compare packed bytes
        results.append(_bin_compare(y_npu, y_gold, "y(fp4 packed)"))
    elif "hifloat8" in y_dtype:
        results.append(_bin_compare(y_npu, y_gold, "y(hifloat8 bits)"))
    elif "float8" in y_dtype:
        results.append(_quant_1lsb_compare(y_npu, y_gold, "y(fp8)"))
    else:
        results.append(_bin_compare(y_npu, y_gold, "y"))

    if is_dynamic:
        results.append(_scale_compare(scale_npu, scale_gold, "scale"))
    else:
        results.append(
            {
                "pass": True,
                "precision": 100.0,
                "error_info": "scaleOut not compared: static mode does not compute it (per aclnnDequantSwigluQuantV2 doc)",
            }
        )
    return results


# ---------------------------------------------------------------------------
# third_party: independent torch composition reference
# ---------------------------------------------------------------------------
class DequantSwigluQuantTorchRef:
    """Independent third-party reference: pure torch composition.

    SiLU/sigmoid/clamp from torch.nn.functional / torch ops; the quantization
    math (per-row amax scaling + saturating cast) is expressed with plain
    torch arithmetic (no reuse of the golden helpers above).
    """

    def __call__(
        self,
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
        import torch
        import torch.nn.functional as F

        def f32(t):
            return None if t is None else t.detach().cpu().float()

        xf = f32(x)
        if xf is None:
            return None
        if isinstance(quant_mode, str):
            dynamic = quant_mode.strip().lower() == "dynamic"
        else:
            dynamic = int(quant_mode) == 1

        # dequant
        if xf.dtype != torch.float32:
            xf = xf.float()
        if (
            torch.tensor(list(xf.shape)).numel()
            and x is not None
            and str(getattr(x, "dtype", "")).endswith("int32")
        ):
            if bias is not None and "int" in str(getattr(bias, "dtype", "")):
                xf = (x.detach().cpu().int() + bias.detach().cpu().int()).float()
            if weight_scale is not None:
                xf = xf * f32(weight_scale)
            if activation_scale is not None:
                xf = xf * f32(activation_scale)
            if bias is not None and "int" not in str(getattr(bias, "dtype", "")):
                xf = xf + f32(bias)
        elif not torch.is_floating_point(x.detach().cpu()):
            xf = xf.float()

        # swiglu (split on the activation axis, moved to last)
        ndim = xf.dim()
        ad = activate_dim + ndim if activate_dim < 0 else int(activate_dim)
        moved = xf.movedim(ad, -1)
        half = moved.shape[-1] // 2
        if swiglu_mode == 1:
            g, lin = moved[..., 0::2], moved[..., 1::2]
        elif activate_left:
            g, lin = moved[..., :half], moved[..., half:]
        else:
            g, lin = moved[..., half:], moved[..., :half]
        if swiglu_mode in (1, 2):
            g = torch.clamp(g, max=clamp_limit)
            lin = torch.clamp(lin, min=-clamp_limit, max=clamp_limit)
            y = g * torch.sigmoid(glu_alpha * g) * (lin + glu_bias)
        elif swiglu_mode == 3:
            y = torch.clamp(F.silu(g), max=clamp_limit) * torch.clamp(
                lin, min=-clamp_limit, max=clamp_limit
            )
        else:
            y = F.silu(g) * lin
        y = y.movedim(-1, ad)

        # quant
        if dynamic:
            row_max = y.abs().amax(dim=-1, keepdim=True)
            smax = SCALAR_MAX_NUM.get(int(dst_type), 127.0)
            # device contract: all-zero rows emit scale = 0 (no guard)
            scale = row_max / smax
            yq = y / scale
            yq = torch.nan_to_num(yq, nan=0.0)
            scale = scale[..., 0].float()
        else:
            qs = f32(quant_scale) if quant_scale is not None else torch.tensor(1.0)
            # device: y / quantScale, plus quantOffset only when that input exists
            yq = y / qs
            if quant_offset is not None:
                yq = yq + f32(quant_offset)
            scale = torch.zeros(yq.shape[:-1], dtype=torch.float32)

        # saturating cast via torch ops
        lo, hi = SAT_RANGE.get(int(dst_type), (-127.0, 127.0))
        yq = torch.clamp(yq, min=lo, max=hi)
        if int(dst_type) == 2:
            rm = {
                "rint": None,
                "round": None,
                "floor": torch.floor,
                "ceil": torch.ceil,
                "trunc": torch.trunc,
            }.get(round_mode)
            if rm is not None:
                yq = rm(yq)
            elif round_mode == "round":
                yq = torch.sign(yq) * torch.floor(torch.abs(yq) + 0.5)
            else:
                yq = torch.round(yq)
            yq = yq.to(torch.int8)
        else:
            yq = yq  # keep fp32; caller adapts the low-bit representation
        return [yq.cpu().numpy(), scale.cpu().numpy()]


# ---------------------------------------------------------------------------
# TestSpec classes (primary registration consumed by TTK)
# ---------------------------------------------------------------------------
class DequantSwigluQuantKernelSpec:
    """Kernel/GEIR spec (numpy I/O). GEIR reuses this spec by op_name."""

    golden = dequant_swiglu_quant_golden
    compare = dequant_swiglu_quant_compare
    customize_inputs = dequant_swiglu_quant_kernel_input
    third_party = {"torch": DequantSwigluQuantTorchRef}
    tolerance = {
        "int8": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float4_e2m1": {"standard": "binary_equal"},
        "float4_e1m2": {"standard": "binary_equal"},
        "hifloat8": {"standard": "binary_equal"},
        "float32": {"standard": "stat_rel_err"},
    }


class AclnnDequantSwigluQuantSpec:
    """ACLNN V1 spec (torch/numpy I/O, C header parameter order)."""

    golden = aclnn_dequant_swiglu_quant_golden
    compare = dequant_swiglu_quant_compare
    third_party = {"torch": DequantSwigluQuantTorchRef}
    tolerance = DequantSwigluQuantKernelSpec.tolerance


class AclnnDequantSwigluQuantV2Spec:
    """ACLNN V2 spec (dstType/roundMode/activateDim/swigluMode variants)."""

    golden = aclnn_dequant_swiglu_quant_v2_golden
    compare = dequant_swiglu_quant_compare
    third_party = {"torch": DequantSwigluQuantTorchRef}
    tolerance = DequantSwigluQuantKernelSpec.tolerance


class DequantSwigluQuantE2ESpec:
    """E2E spec for torch_npu.npu_dequant_swiglu_quant (device representations)."""

    golden = npu_dequant_swiglu_quant_golden
    compare = dequant_swiglu_quant_compare
    third_party = {"torch": DequantSwigluQuantTorchRef}
    tolerance = DequantSwigluQuantKernelSpec.tolerance
