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

Quantization semantics (mirrors kernel and ttk.utilities.mx_quantize):
    - scale_alg=0 (OCP) or FP4 dst types: shared_exp = floor(log2(amax)) - emax;
      the exponent may be negative and is encoded as E8M0 byte = shared_exp + 127
      clipped to [0, 255]; all-zero block -> byte 0 (2^-127); inf/NaN block ->
      NaN (byte 255)
    - scale_alg=1 (cuBLAS, FP8 only): S = amax / max_norm(dst), the E8M0 exponent
      is the fp32 exponent of S rounded up when its mantissa is non-zero
    - elements are quantized on the target dtype grid: the scaled value is split
      into private_exp + mantissa bits, rounded per round_mode, then clipped to
      the max norm of the dst type
    - round modes: rint (ties-to-even), round (ties away from zero, kernel
      CAST_ROUND), floor, ceil, trunc; NaN elements are cast to +0 (mirrors NPU)

mxscale layout (mirrors kernel/infershape):
    - one scale per block_size(=32) block along axis
    - scales are stored in pairs: the axis dim of mxscale is
      ceil(n_blocks / 2) (i.e. block count padded to even) and a last dim
      of 2 is appended; slot k of pair p holds the scale of block (2p + k)
    - all-zero block / padding slot -> scale byte 0 (2^-127)
    - with group_index: axis=-2 packs ceil(rows_g/64) pairs per group
      compactly (allocated axis dim = M // 64 + groupNum); axis=-1 segments
      rows and each row still owns ceil(N/64) pairs; slots not covered by any
      group keep their zero-initialized value (byte 0)

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
    from ml_dtypes import float8_e4m3fn as _fp8_e4m3
    from ml_dtypes import float8_e5m2 as _fp8_e5m2
except ImportError as _exc:
    raise ImportError(
        "swiglu_mx_quant golden requires the ml_dtypes package (pip install ml-dtypes)"
    ) from _exc

try:
    from en_dtypes import float8_e8m0 as _fp8_e8m0
    from en_dtypes import float4_e2m1 as _fp4_e2m1
    from en_dtypes import float4_e1m2 as _fp4_e1m2
except ImportError as _exc:
    raise ImportError(
        "swiglu_mx_quant golden requires the en_dtypes package (pip install en-dtypes)"
    ) from _exc


# dst_type -> (name, numpy dtype, emax, exp_bits, mant_bits, min_exp, max_norm)
# min_exp is the minimum normal exponent of the target format (e1m2 is special-cased to 0).
_DST_TYPE_MAP = {
    40: ("float4_e2m1", _fp4_e2m1, 2, 2, 1, 0, 6.0),
    41: ("float4_e1m2", _fp4_e1m2, 0, 1, 2, 0, 1.75),
    36: ("float8_e4m3fn", _fp8_e4m3, 8, 4, 3, -6, 448.0),
    35: ("float8_e5m2", _fp8_e5m2, 15, 5, 2, -14, 57344.0),
}

_FP32_MIN_NORMAL = np.float32(2.0**-126)
_E8M0_MAX_EXP = 127


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


def _round_mantissa(arr, round_mode):
    """Round on the target grid; 'round' is ties-away-from-zero (kernel CAST_ROUND)."""
    if round_mode in ("rint", "even"):
        return np.rint(arr)
    if round_mode in ("round", "nearest"):
        sign = np.signbit(arr)
        rounded_abs = np.floor(np.abs(arr) + arr.dtype.type(0.5))
        return np.where(sign, -rounded_abs, rounded_abs)
    if round_mode == "floor":
        return np.floor(arr)
    if round_mode == "ceil":
        return np.ceil(arr)
    if round_mode == "trunc":
        return np.trunc(arr)
    raise ValueError(f"Unrecognized round mode: {round_mode}")


def _share_exp_ocp(abs_max, emax):
    """OCP scale: shared_exp = floor(log2(amax)) - emax; all-zero block -> -inf."""
    safe = abs_max + _FP32_MIN_NORMAL * (abs_max == 0)
    share_exp = np.floor(np.log2(safe.astype(np.float32))) - emax
    return np.where(abs_max == 0, -np.inf, share_exp)


def _share_exp_blas(abs_max, max_norm):
    """cuBLAS scale (scale_alg=1, FP8): S = amax / max_norm, the E8M0 exponent is
    the fp32 exponent of S rounded up when its mantissa is non-zero."""
    s_fp32 = (abs_max / np.float32(max_norm)).astype(np.float32)
    bits = s_fp32.view(np.uint32)
    exponents = ((bits & np.uint32(0x7F800000)) >> 23).astype(np.int16)
    mantissas = bits & np.uint32(0x007FFFFF)
    round_up = ((exponents > 0) & (exponents < 254) & (mantissas > 0)) | (
        (exponents == 0) & (mantissas > 2**22)
    )
    exponents = np.where(round_up, exponents + 1, exponents)
    share_exp = (exponents - 127).astype(np.float32)
    return np.where(abs_max == 0, -np.inf, share_exp)


def _quantize_elements(values, share_exp, dst_type, round_mode):
    """Quantize values / 2^share_exp onto the target dtype grid."""
    _, _, _, _, mant_bits, min_exp, max_norm = _DST_TYPE_MAP[dst_type]
    ret = values / np.power(2.0, share_exp)
    private_exp = np.floor(np.log2(np.abs(ret) + (ret == 0)))
    private_exp = np.clip(private_exp, min_exp, None)
    ret = ret / np.power(2.0, private_exp) * float(2**mant_bits)
    ret = _round_mantissa(ret, round_mode)
    ret = ret / float(2**mant_bits) * np.power(2.0, private_exp)
    return np.clip(ret, -max_norm, max_norm)


def _mx_quantize(data_fp32, axis_pos, dst_type, block_size, round_mode, scale_alg):
    """Dynamic MX quantization, returns (quantized_y, mxscale, dst_dtype)."""
    dst_name, dst_dtype, emax, _, _, _, max_norm = _DST_TYPE_MAP[dst_type]
    use_blas = scale_alg != 0 and dst_type in (35, 36)

    shape = list(data_fp32.shape)
    pre_q = _prod(shape[:axis_pos]) if axis_pos > 0 else 1
    q_dim = shape[axis_pos]
    post_q = _prod(shape[axis_pos + 1 :]) if axis_pos + 1 < len(shape) else 1
    flat = data_fp32.reshape(pre_q, q_dim, post_q)

    n_blocks = math.ceil(q_dim / block_size)
    n_pairs = (n_blocks + 1) // 2
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

        # Scale exponent: OCP or cuBLAS; E8M0 encoding clamps to [-127, 127],
        # above that (inf/huge block) -> NaN (byte 255).
        if use_blas:
            share_exp = _share_exp_blas(abs_max, max_norm)
        else:
            share_exp = _share_exp_ocp(abs_max, emax)
        share_exp = np.where(share_exp > _E8M0_MAX_EXP, np.float32(np.nan), share_exp)
        share_exp = np.where(
            share_exp < -_E8M0_MAX_EXP, np.float32(-_E8M0_MAX_EXP), share_exp
        )

        scaled_q = _quantize_elements(chunk, share_exp, dst_type, round_mode)

        actual_len = end - start
        y_flat[:, start:end, :] = scaled_q[:, :actual_len, :]
        scale_flat[:, b, :] = np.power(2.0, share_exp)[:, 0, :]

    # NPU casts NaN elements (NaN-scale blocks) to +0
    y_out = np.nan_to_num(y_flat.reshape(shape), nan=0.0, copy=False)

    # Pack scales in pairs: axis dim becomes n_pairs, last dim of 2 appended;
    # slot k of pair p holds the scale of block (2p + k); padding slots hold
    # 2^-127 (byte 0), mirroring the kernel's zero-padded phantom block.
    padded = np.full((pre_q, n_pairs * 2, post_q), np.float32(2.0**-127))
    padded[:, :n_blocks, :] = scale_flat
    pairs = padded.reshape(pre_q, n_pairs, 2, post_q)
    lead = tuple(shape[:axis_pos])
    trail = tuple(shape[axis_pos + 1 :])
    tmp = pairs.reshape(lead + (n_pairs, 2) + trail)
    k_axis = len(lead) + 1
    perm = [i for i in range(tmp.ndim) if i != k_axis] + [k_axis]
    scale_out = np.ascontiguousarray(np.transpose(tmp, perm))

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
        swiglu_result = swiglu_result.astype(_bf16)
    elif "float16" in str(x.dtype):
        swiglu_result = swiglu_result.astype(np.float16)

    swiglu_fp32 = swiglu_result.astype(np.float32)

    if group_index is not None:
        y_shape = list(swiglu_fp32.shape)
        ndim = len(y_shape)
        scale_shape = list(y_shape)
        if axis_pos == ndim - 2:
            # axis=-2 with group_index (x must be 2D):
            # allocated pairs = M // 64 + groupNum (mirrors infershape)
            scale_shape[axis_pos] = y_shape[axis_pos] // (block_size * 2) + len(
                group_index
            )
        else:
            scale_shape[axis_pos] = math.ceil(y_shape[axis_pos] / (block_size * 2))
        scale_shape.append(2)

        y_dtype = _DST_TYPE_MAP[dst_type][1]
        y = np.zeros(y_shape, dtype=y_dtype)
        scale = np.zeros(scale_shape, dtype=_fp8_e8m0)

        if axis_pos == ndim - 2:
            # axis=-2: each group owns ceil(rows_g / 64) scale pairs, packed compactly
            pair_off = 0
            start = 0
            for gv in group_index:
                gv = int(gv)
                if gv <= 0:
                    continue
                y_part, scale_part, _ = _mx_quantize(
                    swiglu_fp32[start : start + gv],
                    axis_pos,
                    dst_type,
                    block_size,
                    round_mode,
                    scale_alg,
                )
                y[start : start + gv] = y_part.astype(y_dtype)
                npair_g = scale_part.shape[axis_pos]
                idx = (slice(None),) * axis_pos + (slice(pair_off, pair_off + npair_g),)
                scale[idx] = scale_part.astype(_fp8_e8m0)
                pair_off += npair_g
                start += gv
        else:
            # axis=-1: group_index segments the flattened rows of the SwiGLU output
            rows = swiglu_fp32.reshape(-1, y_shape[-1])
            y_rows = y.reshape(-1, y_shape[-1])
            scale_rows = scale.reshape(-1, scale_shape[-2], 2)
            start = 0
            for gv in group_index:
                gv = int(gv)
                if gv <= 0:
                    continue
                y_part, scale_part, _ = _mx_quantize(
                    rows[start : start + gv],
                    rows.ndim - 1,
                    dst_type,
                    block_size,
                    round_mode,
                    scale_alg,
                )
                y_rows[start : start + gv] = y_part.astype(y_dtype)
                scale_rows[start : start + gv] = scale_part.astype(_fp8_e8m0)
                start += gv
    else:
        y_np, scale_np, dst_dtype = _mx_quantize(
            swiglu_fp32, axis_pos, dst_type, block_size, round_mode, scale_alg
        )
        y = y_np.astype(dst_dtype)
        scale = scale_np.astype(_fp8_e8m0)

    return [y, scale]


__golden__ = {"kernel": {"swiglu_mx_quant": "__golden_swiglu_mx_quant"}}
