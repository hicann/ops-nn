#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK golden / TestSpec for swiglu_mx_quant — Kernel / GEIR / ACLNN / E2E.

Math core (torch, CPU fp32):
    swiglu : mode0 SiLU(gate)*linear; mode1 odd/even clamp variant; mode2 half
             split clamp variant; mode3 SiLU-then-clamp variant. The SwiGLU
             intermediate is rounded back to the input precision (fp16/bf16)
             before quantization, mirroring the kernel's intermediate storage.
    mx     : per 32-block dynamic MX quantization. OCP scale
             (floor(log2(amax)) - emax) or cuBLAS scale (amax/max_norm with
             mantissa-aware exponent round-up, FP8 only). E8M0 exponent bytes,
             pair packing (slot k of pair p = block 2p+k) and -2 axis
             interleaving are layout/encoding steps kept in numpy bit ops.

Device-side output representations (torch_npu, verified on Ascend950PR):
    y       : FP8 -> torch native fp8 dtypes; FP4 -> packed uint8 (even elem
              low nibble, last dim halved)
    mxscale : uint8 bit-view of FLOAT8_E8M0

E2E binding note: the local op-plugin build only accepts swiglu_mode 0/1 for
torch_npu.npu_swiglu_mx_quant (RuntimeError "swiglu_mode only support 0 or
1"); modes 2/3 are exercised through the ACLNN/Kernel paths.
"""

import math

import numpy as np

try:
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

__golden__ = {
    "kernel": {"swiglu_mx_quant": "__golden_swiglu_mx_quant"},
    "aclnn": {"aclnnSwigluMxQuant": "aclnn_swiglu_mx_quant_golden"},
    "e2e": {"torch_npu.npu_swiglu_mx_quant": "npu_swiglu_mx_quant_golden"},
}

__input__ = {
    "aclnn": {"aclnnSwigluMxQuant": "swiglu_mx_quant_group_index_input"},
    "e2e": {"torch_npu.npu_swiglu_mx_quant": "swiglu_mx_quant_group_index_input"},
}

__spec__ = {
    "swiglu_mx_quant": "SwigluMxQuantKernelSpec",
    "aclnnSwigluMxQuant": "AclnnSwigluMxQuantSpec",
    "torch_npu.npu_swiglu_mx_quant": "SwigluMxQuantE2ESpec",
}

# dst_type -> (name, numpy dtype, emax, exp_bits, mant_bits, min_exp, max_norm)
_DST_TYPE_MAP = {
    40: ("float4_e2m1", _fp4_e2m1, 2, 2, 1, 0, 6.0),
    41: ("float4_e1m2", _fp4_e1m2, 0, 1, 2, 0, 1.75),
    36: ("float8_e4m3fn", _fp8_e4m3, 8, 4, 3, -6, 448.0),
    35: ("float8_e5m2", _fp8_e5m2, 15, 5, 2, -14, 57344.0),
}

# torch ScalarType enum (E2E dst_type) -> CANN dst_type code
TORCH_DTYPE_ENUM_TO_DST_TYPE = {
    23: 35,  # float8_e5m2
    24: 36,  # float8_e4m3fn
    291: 35,  # float8_e5m2 (npu alias)
    292: 36,  # float8_e4m3fn (npu alias)
    296: 40,  # float4_e2m1 (npu packed)
    297: 41,  # float4_e1m2 (npu packed)
}

_FP32_MIN_NORMAL = np.float32(2.0**-126)
_E8M0_MAX_EXP = 127
BLOCK_SIZE = 32


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p


# ---------------------------------------------------------------------------
# Torch math core
# ---------------------------------------------------------------------------
def _torch_sigmoid(x):
    import torch

    return torch.sigmoid(x)


def _torch_silu(x):
    import torch.nn.functional as F

    return F.silu(x)


def _torch_round_mode(v, round_mode):
    """Apply the round mode on the mantissa grid (torch)."""
    import torch

    if round_mode in ("rint", "even"):
        return torch.round(v)
    if round_mode in ("round", "nearest"):
        # signbit keeps the sign of -0.0 (device keeps -0 in fp4 encodings)
        neg = torch.signbit(v)
        rounded_abs = torch.floor(torch.abs(v) + 0.5)
        return torch.where(neg, -rounded_abs, rounded_abs)
    if round_mode == "floor":
        return torch.floor(v)
    if round_mode == "ceil":
        return torch.ceil(v)
    if round_mode == "trunc":
        return torch.trunc(v)
    raise ValueError(f"Unrecognized round mode: {round_mode}")


def _torch_swiglu(
    x_t,
    dim_pos,
    swiglu_mode,
    activate_left,
    clamp_limit,
    glu_alpha,
    glu_bias,
    in_dtype_name,
):
    """SwiGLU activation (modes 0-3) in torch fp32, rounded back to the input
    precision before MX quantization (kernel intermediate storage contract)."""
    import torch

    pre = _prod(x_t.shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(x_t.shape[dim_pos:])
    xf = x_t.reshape(pre, cut)
    h = cut // 2

    if swiglu_mode == 0:
        a = xf[:, :h]
        b = xf[:, h:]
        if activate_left:
            res = _torch_silu(a) * b
        else:
            res = _torch_silu(b) * a
    elif swiglu_mode == 1:
        a = xf[:, 0::2]
        b = xf[:, 1::2]
        a = torch.clamp(a, max=clamp_limit)
        b = torch.clamp(b, min=-clamp_limit, max=clamp_limit)
        res = a * _torch_sigmoid(glu_alpha * a) * (b + glu_bias)
    elif swiglu_mode == 2:
        a = xf[:, :h]
        b = xf[:, h:]
        if not activate_left:
            a, b = b, a
        a = torch.clamp(a, max=clamp_limit)
        b = torch.clamp(b, min=-clamp_limit, max=clamp_limit)
        res = a * _torch_sigmoid(glu_alpha * a) * (b + glu_bias)
    elif swiglu_mode == 3:
        a = xf[:, :h]
        b = xf[:, h:]
        if not activate_left:
            a, b = b, a
        a = torch.clamp(_torch_silu(a), max=clamp_limit)
        b = torch.clamp(b, min=-clamp_limit, max=clamp_limit)
        res = a * b
    else:
        raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")

    # round the SwiGLU intermediate back to the input precision
    if in_dtype_name == "bfloat16":
        res = res.to(torch.bfloat16).to(torch.float32)
    elif in_dtype_name == "float16":
        res = res.to(torch.float16).to(torch.float32)

    out_shape = list(x_t.shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    return res.reshape(out_shape)


def _torch_share_exp_ocp(amax):
    """OCP shared exponent: floor(log2(amax)) - emax; zero block -> -inf."""
    import torch

    dst_type = _torch_share_exp_ocp._dst_type

    emax = _DST_TYPE_MAP[dst_type][2]
    safe = torch.where(amax == 0, torch.full_like(amax, _FP32_MIN_NORMAL), amax)
    share_exp = torch.floor(torch.log2(safe)) - emax
    return torch.where(amax == 0, torch.full_like(share_exp, float("-inf")), share_exp)


def _torch_share_exp_blas(amax):
    """cuBLAS shared exponent (scale_alg=1, FP8 only): S = amax/max_norm, the
    E8M0 exponent is the fp32 exponent of S rounded up when its mantissa is
    non-zero."""
    import torch

    dst_type = _torch_share_exp_blas._dst_type
    max_norm = _DST_TYPE_MAP[dst_type][6]
    s_fp32 = (amax / max_norm).to(torch.float32)
    bits = s_fp32.view(torch.int32)
    exponents = ((bits & 0x7F800000) >> 23).to(torch.int16)
    mantissas = bits & 0x007FFFFF
    round_up = ((exponents > 0) & (exponents < 254) & (mantissas > 0)) | (
        (exponents == 0) & (mantissas > 2**22)
    )
    exponents = torch.where(round_up, exponents + 1, exponents)
    share_exp = (exponents - 127).to(torch.float32)
    return torch.where(amax == 0, torch.full_like(share_exp, float("-inf")), share_exp)


def _torch_quantize_elements(values, share_exp, dst_type, round_mode):
    """Quantize values / 2^share_exp onto the target dtype grid (torch)."""
    import torch

    _, _, _, _, mant_bits, min_exp, max_norm = _DST_TYPE_MAP[dst_type]
    ret = values / torch.pow(2.0, share_exp)
    private_exp = torch.floor(torch.log2(torch.abs(ret) + (ret == 0).to(ret.dtype)))
    private_exp = torch.clamp(private_exp, min=min_exp)
    ret = ret / torch.pow(2.0, private_exp) * float(2**mant_bits)
    ret = _torch_round_mode(ret, round_mode)
    ret = ret / float(2**mant_bits) * torch.pow(2.0, private_exp)
    return torch.clamp(ret, min=-max_norm, max=max_norm)


def _torch_mx_quantize(data_t, axis_pos, dst_type, block_size, round_mode, scale_alg):
    """Torch math core of the MX block quantization.

    Block amax / shared exponent / element quantization run in torch; the E8M0
    byte encoding and the pair/interleave scale layout are bit-level encoding
    steps done in numpy on the way out. Returns (y_t fp32 torch, scale_np
    uint8 numpy, dst_np_dtype).
    """
    import torch

    use_blas = scale_alg != 0 and dst_type in (35, 36)
    shape = list(data_t.shape)
    pre_q = _prod(shape[:axis_pos]) if axis_pos > 0 else 1
    q_dim = shape[axis_pos]
    post_q = _prod(shape[axis_pos + 1 :]) if axis_pos + 1 < len(shape) else 1
    flat = data_t.reshape(pre_q, q_dim, post_q)

    n_blocks = math.ceil(q_dim / block_size)
    n_pairs = (n_blocks + 1) // 2
    pad_len = n_blocks * block_size - q_dim
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, 0, 0, pad_len))
    blocks = flat.reshape(pre_q, n_blocks, block_size, post_q)

    amax = torch.amax(torch.abs(blocks), dim=2, keepdim=True)  # (pre, nb, 1, post)

    _torch_share_exp_ocp._dst_type = dst_type
    _torch_share_exp_blas._dst_type = dst_type
    if use_blas:
        share_exp = _torch_share_exp_blas(amax)
    else:
        share_exp = _torch_share_exp_ocp(amax)
    share_exp = torch.where(
        share_exp > _E8M0_MAX_EXP,
        torch.full_like(share_exp, float("nan")),
        share_exp,
    )
    share_exp = torch.clamp(share_exp, min=-_E8M0_MAX_EXP)

    q = _torch_quantize_elements(blocks, share_exp, dst_type, round_mode)
    q = torch.nan_to_num(q, nan=0.0)  # NPU casts NaN elements to +0
    y_t = q.reshape(pre_q, n_blocks * block_size, post_q)[:, :q_dim, :].reshape(shape)

    # ---- E8M0 encoding + pair/interleave layout (numpy bit ops) ----
    se_np = share_exp[:, :, 0, :].cpu().numpy().astype(np.float32)
    # NaN exponent (overflow block) encodes as byte 255
    scale_bytes = np.where(
        np.isnan(se_np), np.uint8(255), np.clip(se_np + 127.0, 0, 255)
    ).astype(np.uint8)

    # pad blocks to even count with byte 0 (2^-127), pair pack along axis
    padded = np.full((pre_q, n_pairs * 2, post_q), 0, dtype=np.uint8)
    padded[:, :n_blocks, :] = scale_bytes
    pairs = padded.reshape(pre_q, n_pairs, 2, post_q)
    lead = tuple(shape[:axis_pos])
    trail = tuple(shape[axis_pos + 1 :])
    tmp = pairs.reshape(lead + (n_pairs, 2) + trail)
    k_axis = len(lead) + 1
    perm = [i for i in range(tmp.ndim) if i != k_axis] + [k_axis]
    scale_out = np.ascontiguousarray(np.transpose(tmp, perm))
    return y_t, scale_out, _DST_TYPE_MAP[dst_type][1]


def _torch_grouped_axis_last2(
    swiglu_t, group_index, axis_pos, dst_type, block_size, round_mode, scale_alg
):
    """axis=-2 with group_index: per-group quantization with compact pair
    packing (allocated axis dim = M//64 + groupNum)."""
    import torch

    y_shape = list(swiglu_t.shape)
    y_t = torch.zeros_like(swiglu_t)
    pair_rows = y_shape[axis_pos] // (block_size * 2) + len(group_index)
    scale_np = np.zeros(
        tuple(y_shape[:axis_pos])
        + (pair_rows,)
        + tuple(y_shape[axis_pos + 1 :])
        + (2,),
        dtype=np.uint8,
    )
    pair_off = 0
    start = 0
    for gv in group_index:
        gv = int(gv)
        if gv <= 0:
            continue
        y_part, scale_part, _ = _torch_mx_quantize(
            swiglu_t[start : start + gv],
            axis_pos,
            dst_type,
            block_size,
            round_mode,
            scale_alg,
        )
        y_t[start : start + gv] = y_part
        npair_g = scale_part.shape[axis_pos]
        idx = (slice(None),) * axis_pos + (slice(pair_off, pair_off + npair_g),)
        scale_np[idx] = scale_part
        pair_off += npair_g
        start += gv
    return y_t, scale_np


def _torch_grouped_axis_last(
    swiglu_t, group_index, dst_type, block_size, round_mode, scale_alg
):
    """axis=-1 with group_index: row segmentation, each row owns ceil(N/64)
    pairs; uncovered rows keep zero-initialized scales."""
    import torch

    y_shape = list(swiglu_t.shape)
    rows = swiglu_t.reshape(-1, y_shape[-1])
    y_t = torch.zeros_like(rows)
    n_pairs = math.ceil(y_shape[-1] / (block_size * 2))
    scale_np = np.zeros((rows.shape[0], n_pairs, 2), dtype=np.uint8)
    start = 0
    for gv in group_index:
        gv = int(gv)
        if gv <= 0:
            continue
        y_part, scale_part, _ = _torch_mx_quantize(
            rows[start : start + gv],
            rows.dim() - 1,
            dst_type,
            block_size,
            round_mode,
            scale_alg,
        )
        y_t[start : start + gv] = y_part
        scale_np[start : start + gv] = scale_part
        start += gv
    # The device keeps the N-D leading shape of y for the scale output; the
    # row loop above works on the flattened row view, so restore it here.
    scale_shape = tuple(y_shape[:-1]) + (n_pairs, 2)
    return y_t.reshape(y_shape), scale_np.reshape(scale_shape)


def _torch_swiglu_mx_quant(x, group_index, kwargs):
    """Full torch chain: returns (y_t fp32, scale_np uint8, dst_np_dtype)."""
    import torch

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
    block_size = BLOCK_SIZE

    def to_f32(v):
        if v is None:
            return None
        if hasattr(v, "detach"):
            t = v.detach().cpu()
            if t.dtype == torch.bfloat16:
                return t.to(torch.float32)
            if not torch.is_floating_point(t):
                return t.to(torch.float32)
            return t.to(torch.float32)
        arr = np.asarray(v)
        return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))

    x_t = to_f32(x)
    in_dtype_name = ""
    if hasattr(x, "dtype"):
        in_dtype_name = str(getattr(x, "dtype"))

    ndim = x_t.dim()
    dim_pos = activate_dim % ndim
    axis_pos = axis % ndim

    swiglu_t = _torch_swiglu(
        x_t,
        dim_pos,
        swiglu_mode,
        activate_left,
        clamp_limit,
        glu_alpha,
        glu_bias,
        in_dtype_name.replace("torch.", ""),
    )

    if group_index is not None:
        gi = np.asarray(group_index).reshape(-1).astype(np.int64)
        if axis_pos == ndim - 2:
            y_t, scale_np = _torch_grouped_axis_last2(
                swiglu_t, gi, axis_pos, dst_type, block_size, round_mode, scale_alg
            )
        else:
            y_t, scale_np = _torch_grouped_axis_last(
                swiglu_t, gi, dst_type, block_size, round_mode, scale_alg
            )
    else:
        y_t, scale_np, _ = _torch_mx_quantize(
            swiglu_t, axis_pos, dst_type, block_size, round_mode, scale_alg
        )
    return y_t, scale_np, _DST_TYPE_MAP[dst_type][1]


def _pack_fp4_to_uint8(ele):
    """even element -> low nibble, odd element -> high nibble (device pack)."""
    if not isinstance(ele, np.ndarray) or "float4" not in str(ele.dtype):
        return ele
    if ele.shape[-1] % 2:
        raise ValueError("FP4 packing requires an even last dim")
    raw = ele.view(np.uint8) & 0x0F
    pairs = raw.reshape(raw.shape[:-1] + (-1, 2))
    packed = pairs[..., 0].astype(np.uint16) | (pairs[..., 1].astype(np.uint16) << 4)
    return packed.astype(np.uint8)


def _y_to_device_repr(y_np, dst_np_dtype, dst_type):
    """Convert the quantized y to the device-side representation per path."""
    y_q = y_np.astype(dst_np_dtype)
    if dst_type in (40, 41):
        return _pack_fp4_to_uint8(y_q)
    return y_q


# ---------------------------------------------------------------------------
# Kernel / GEIR golden (numpy in, numpy out)
# ---------------------------------------------------------------------------
def __golden_swiglu_mx_quant(*input_arrays, **kwargs):
    """Kernel golden for swiglu_mx_quant (numpy I/O, op-def parameter order)."""
    x = np.asarray(input_arrays[0])
    group_index = None
    if len(input_arrays) > 1 and input_arrays[1] is not None:
        group_index = np.asarray(input_arrays[1])
    y_t, scale_np, dst_np_dtype = _torch_swiglu_mx_quant(x, group_index, kwargs)
    y_np = y_t.cpu().numpy()
    # kernel/ACLNN paths expose flat output dtypes and TTK unpacks FP4 itself
    # scale bytes are bit-reinterpreted (view), not value-cast, to e8m0
    return [y_np.astype(dst_np_dtype), scale_np.view(_fp8_e8m0)]


# ---------------------------------------------------------------------------
# ACLNN golden (C header parameter order; torch.Tensor or numpy in)
# ---------------------------------------------------------------------------
def aclnn_swiglu_mx_quant_golden(
    x,
    groupIndexOptional=None,
    activateDim=-1,
    activateLeft=False,
    swigluMode=0,
    clampLimit=7.0,
    gluAlpha=1.702,
    gluBias=1.0,
    groupMode=0,
    axis=-1,
    dstType=40,
    roundModeOptional="rint",
    scaleAlg=0,
    maxDtypeValue=0.0,
    yOut=None,
    mxscaleOut=None,
    **kwargs,
):
    """Golden for aclnnSwigluMxQuant (args follow GetWorkspaceSize minus
    workspaceSize/executor)."""
    kwargs = dict(kwargs)
    kwargs.update(
        activate_dim=int(activateDim),
        activate_left=bool(activateLeft),
        swiglu_mode=int(swigluMode),
        clamp_limit=float(clampLimit),
        glu_alpha=float(gluAlpha),
        glu_bias=float(gluBias),
        axis=int(axis),
        dst_type=int(dstType),
        round_mode=str(roundModeOptional) if roundModeOptional is not None else "rint",
        scale_alg=int(scaleAlg),
    )
    y_t, scale_np, dst_np_dtype = _torch_swiglu_mx_quant(x, groupIndexOptional, kwargs)
    y_np = y_t.cpu().numpy()
    # scale bytes are bit-reinterpreted (view), not value-cast, to e8m0
    return [y_np.astype(dst_np_dtype), scale_np.view(_fp8_e8m0)]


# ---------------------------------------------------------------------------
# E2E golden (torch_npu signature; device-side representations)
# ---------------------------------------------------------------------------
def npu_swiglu_mx_quant_golden(
    x,
    group_index=None,
    *,
    activate_dim=-1,
    activate_left=False,
    swiglu_mode=0,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
    group_mode=0,
    axis=-1,
    dst_type=296,
    round_mode="rint",
    scale_alg=0,
    max_dtype_value=0.0,
    **kwargs,
):
    """E2E golden for torch_npu.npu_swiglu_mx_quant.

    dst_type arrives as a torch ScalarType enum (default 296 = FP4_E2M1).
    Returns y in the device representation (fp8 native dtype / FP4 packed
    uint8) and mxscale as the uint8 bit-view of E8M0.
    """
    dst = TORCH_DTYPE_ENUM_TO_DST_TYPE.get(int(dst_type), 40)
    kw = dict(
        activate_dim=activate_dim,
        activate_left=activate_left,
        swiglu_mode=swiglu_mode,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
        axis=axis,
        dst_type=dst,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )
    y_t, scale_np, dst_np_dtype = _torch_swiglu_mx_quant(x, group_index, kw)
    y_np = y_t.cpu().numpy()
    y_dev = _y_to_device_repr(y_np, dst_np_dtype, dst)
    return [np.ascontiguousarray(y_dev), np.ascontiguousarray(scale_np)]


# ---------------------------------------------------------------------------
# Input normalization (group_index hygiene)
# ---------------------------------------------------------------------------
def swiglu_mx_quant_group_index_input(x, group_index=None, *extra, **kwargs):
    """Sort/clamp group_index in place and align the last element with the
    total row count of the SwiGLU output (kernel contract)."""
    if group_index is None:
        return
    xshape = tuple(x.shape) if hasattr(x, "shape") else np.asarray(x).shape
    swiglu_rows = int(np.prod(xshape[:-1]))
    if hasattr(group_index, "copy_"):
        import torch

        group_index.copy_(torch.sort(group_index).values)
        group_index.clamp_(0, swiglu_rows)
        group_index[-1] = swiglu_rows
    else:
        arr = np.asarray(group_index)
        arr2 = np.sort(arr)
        arr2 = np.clip(arr2, 0, swiglu_rows)
        arr2[-1] = swiglu_rows
        group_index[...] = arr2


# ---------------------------------------------------------------------------
# Custom compare: per-output, per-dtype standards
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


def _bin_compare(actual, expected, output_name):
    ar, er = _raw_uint8(_ensure_np(actual)), _raw_uint8(_ensure_np(expected))
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


def _mxscale_valid_pair_rows(group_index, block_size=32):
    """Compact pair-row indices on the packed-pair axis covered by the groups.

    Only meaningful for a grouped NON-LAST quantization axis (the packed pair
    axis is the quantization axis position in the scale layout): group g owns
    ceil(rows_g / (block_size * 2)) pairs placed compactly; zero/negative
    groups own nothing (mirrors the golden grouped layout).
    """
    gi = np.asarray(group_index).reshape(-1).astype(np.int64)
    rows = []
    pair_off = 0
    for gv in gi:
        gv = int(gv)
        if gv <= 0:
            continue
        npair_g = math.ceil(gv / (block_size * 2))
        rows.extend(range(pair_off, pair_off + npair_g))
        pair_off += npair_g
    return np.asarray(rows, dtype=np.int64)


def _mx_element_compare(actual, expected, output_name):
    """Quantized y elements: FP8 uses a 1-LSB (requant) tolerance; FP4 packed
    uint8 and other dtypes compare binary."""
    a = _ensure_np(actual)
    e = _ensure_np(expected)
    if a.size != e.size:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{output_name} size mismatch: {a.size} vs {e.size}",
        }
    dtype_name = f"{a.dtype} {e.dtype}".lower()
    if "float8_e4m3" not in dtype_name and "float8_e5m2" not in dtype_name:
        return _bin_compare(a, e, output_name)

    af = a.reshape(-1).astype(np.float32)
    ef = e.reshape(-1).astype(np.float32)
    if "float8_e4m3" in dtype_name:
        lsb_rel = 2.0**-3
    else:
        lsb_rel = 2.0**-2
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


def swiglu_mx_quant_compare(
    y_npu, mxscale_npu, y_gold, mxscale_gold, compare_context=None, **kwargs
):
    """Per-output compare for swiglu_mx_quant (all paths).

    Standards: y fp8 -> requant (|err| <= 1 LSB); y fp4 packed -> binary;
    mxscale E8M0 bytes -> binary over ALL valid data:
      - un-grouped: the whole tensor (every leading index x every packed
        pair; odd-block padding slots are contract zeros on both sides);
      - grouped last-axis: the first `covered` flattened leading rows (each
        with all its pairs); trailing uncovered rows are garbage by contract;
      - grouped non-last-axis: the compact pair chain covered by the groups
        on the pair axis (quantization-axis position in the scale layout).
    """
    y_shape = None
    attrs = {}
    axis_pos = None
    if compare_context is not None:
        attrs = dict(compare_context.attributes or {})
        if compare_context.input_tensors is not None and len(
            compare_context.input_tensors
        ):
            xt = compare_context.input_tensors[0]
            x_shape = (
                tuple(np.asarray(xt).shape)
                if not hasattr(xt, "shape")
                else tuple(xt.shape)
            )
            ad = int(attrs.get("activateDim", attrs.get("activate_dim", -1)))
            ax = int(attrs.get("axis", -1))
            ndim = len(x_shape)
            dim_pos = ad % ndim
            axis_pos = ax % ndim
            ys = list(x_shape)
            ys[dim_pos] = ys[dim_pos] // 2
            y_shape = tuple(ys)
    if y_shape is None:
        y_shape = tuple(_ensure_np(y_gold).shape)
    if axis_pos is None:
        axis_pos = len(y_shape) - 1  # default axis=-1

    gi = None
    if compare_context is not None and compare_context.input_tensors is not None:
        its = compare_context.input_tensors
        if len(its) > 1 and its[1] is not None:
            gi = np.asarray(its[1]).reshape(-1).astype(np.int64)

    results = [_mx_element_compare(y_npu, y_gold, "y")]

    a_scale = _ensure_np(mxscale_npu)
    e_scale = _ensure_np(mxscale_gold)
    if a_scale.shape != e_scale.shape:
        results.append(
            {
                "pass": False,
                "precision": 0.0,
                "error_info": (
                    f"mxscale shape mismatch: {a_scale.shape} vs {e_scale.shape}"
                ),
            }
        )
        return results

    # Scale layout: (leading..., n_pairs_on_quant_axis, trailing..., 2) — the
    # packed-pair axis sits at the quantization-axis position of y (axis=-1
    # -> second-to-last dim of the scale; non-last axis -> that axis dim).
    pair_axis = axis_pos
    if gi is None:
        # Un-grouped: every leading index owns its full packed-pair row set;
        # odd-block padding slots are contract zeros on both sides. Compare
        # the whole tensor (fixes the missed leading rows for axis=-1 where
        # the pair axis is NOT dim 0).
        results.append(_bin_compare(a_scale, e_scale, "mxscale"))
    elif pair_axis == len(y_shape) - 1:
        # Grouped last-axis quantization: groups segment the FLATTENED
        # leading rows; the first `covered` rows are valid, trailing rows are
        # zero-initialized on the golden side and garbage on the device side
        # (contract) — compare only the covered rows, all pairs per row.
        total_rows = int(np.prod(y_shape[:-1])) if len(y_shape) > 1 else 1
        covered = int(min(np.clip(gi, 0, None).sum(), total_rows))
        flat_tail = a_scale.shape[len(y_shape) - 1 :]
        flat_a = a_scale.reshape((-1,) + flat_tail)
        flat_e = e_scale.reshape((-1,) + flat_tail)
        if covered >= total_rows:
            results.append(_bin_compare(a_scale, e_scale, "mxscale"))
        else:
            results.append(
                _bin_compare(flat_a[:covered], flat_e[:covered], "mxscale(valid rows)")
            )
    else:
        # Grouped non-last-axis quantization (2D per doc): compact pair chain
        # on the pair axis; only the covered pairs are compared.
        valid = _mxscale_valid_pair_rows(gi)
        if valid.size and int(valid[-1]) < a_scale.shape[pair_axis]:
            take = [slice(None)] * a_scale.ndim
            take[pair_axis] = valid
            sel = tuple(take)
            results.append(
                _bin_compare(a_scale[sel], e_scale[sel], "mxscale(valid rows)")
            )
        else:
            results.append(_bin_compare(a_scale, e_scale, "mxscale"))
    return results


# ---------------------------------------------------------------------------
# third_party: independent torch composition reference
# ---------------------------------------------------------------------------
class SwigluMxQuantTorchRef:
    """Independent third-party reference: pure torch composition.

    torch.nn.functional.silu/sigmoid/clamp for the activation; the MX block
    scale/element quantization expressed with plain torch ops (amax, log2,
    floor, pow, round) — no reuse of the golden helpers above.
    """

    def __call__(
        self,
        x,
        group_index=None,
        *,
        activate_dim=-1,
        activate_left=False,
        swiglu_mode=0,
        clamp_limit=7.0,
        glu_alpha=1.702,
        glu_bias=1.0,
        axis=-1,
        dst_type=40,
        round_mode="rint",
        scale_alg=0,
        **kwargs,
    ):
        import torch
        import torch.nn.functional as F

        dst = TORCH_DTYPE_ENUM_TO_DST_TYPE.get(dst_type, dst_type)
        dst = int(dst)
        _, _, emax, _, mant_bits, min_exp, max_norm = _DST_TYPE_MAP[dst]

        xf = x.detach().cpu()
        in_prec = str(xf.dtype).replace("torch.", "")
        xf = xf.float()
        ndim = xf.dim()
        dim_pos = activate_dim % ndim
        moved = xf.movedim(dim_pos, -1)
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
        if in_prec == "bfloat16":
            y = y.to(torch.bfloat16).float()
        elif in_prec == "float16":
            y = y.to(torch.float16).float()
        y = y.movedim(-1, dim_pos)

        axis_pos = axis % ndim
        # block quantization on the magnitude grid
        q_dim = y.shape[axis_pos]
        moved_q = y.movedim(axis_pos, 0)
        n_blocks = (q_dim + 31) // 32
        pad = n_blocks * 32 - q_dim
        if pad:
            moved_q = F.pad(moved_q, (0, 0) * (ndim - 1) + (0, pad))
        blocks = moved_q.reshape(n_blocks, 32, *moved_q.shape[1:])
        amax = blocks.abs().amax(dim=1, keepdim=True)
        if scale_alg == 1 and dst in (35, 36):
            s = amax / max_norm
            e = torch.floor(torch.log2(torch.where(s > 0, s, torch.ones_like(s))))
            mant = s / torch.pow(2.0, e)
            share = torch.where(mant > 1.0, e + 1, e)
        else:
            safe = torch.where(amax > 0, amax, torch.ones_like(amax))
            share = torch.floor(torch.log2(safe)) - emax
        share = share.clamp(min=-127, max=127)
        v = blocks / torch.pow(2.0, share)
        pe = torch.floor(torch.log2(torch.abs(v) + (v == 0).float())).clamp(min=min_exp)
        v = v / torch.pow(2.0, pe) * float(2**mant_bits)
        if round_mode in ("round", "nearest"):
            neg = torch.signbit(v)
            v = torch.where(
                neg, -torch.floor(torch.abs(v) + 0.5), torch.floor(torch.abs(v) + 0.5)
            )
        else:
            v = getattr(torch, round_mode if round_mode != "rint" else "round")(v)
        v = v / float(2**mant_bits) * torch.pow(2.0, pe)
        v = torch.clamp(v, min=-max_norm, max=max_norm)
        v = v.reshape(n_blocks * 32, *moved_q.shape[1:])[:q_dim]
        y_out = v.movedim(0, axis_pos)
        return [y_out.numpy().astype(_DST_TYPE_MAP[dst][1])]


# ---------------------------------------------------------------------------
# TestSpec classes
# ---------------------------------------------------------------------------
_kernel_golden_fn = __golden_swiglu_mx_quant  # avoid class-body name mangling


class SwigluMxQuantKernelSpec:
    """Kernel/GEIR spec (numpy I/O)."""

    golden = _kernel_golden_fn
    compare = swiglu_mx_quant_compare
    third_party = {"torch": SwigluMxQuantTorchRef}
    tolerance = {
        "float8_e4m3fn": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float4_e2m1": {"standard": "binary_equal"},
        "float4_e1m2": {"standard": "binary_equal"},
        "float8_e8m0": {"standard": "binary_equal"},
    }


class AclnnSwigluMxQuantSpec:
    """ACLNN spec (torch/numpy I/O, C header parameter order)."""

    golden = aclnn_swiglu_mx_quant_golden
    compare = swiglu_mx_quant_compare
    third_party = {"torch": SwigluMxQuantTorchRef}
    tolerance = SwigluMxQuantKernelSpec.tolerance


class SwigluMxQuantE2ESpec:
    """E2E spec for torch_npu.npu_swiglu_mx_quant (device representations)."""

    golden = npu_swiglu_mx_quant_golden
    compare = swiglu_mx_quant_compare
    third_party = {"torch": SwigluMxQuantTorchRef}
    tolerance = SwigluMxQuantKernelSpec.tolerance
