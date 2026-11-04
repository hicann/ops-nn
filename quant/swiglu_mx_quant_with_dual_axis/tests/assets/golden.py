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
"""
TTK golden / TestSpec for swiglu_mx_quant_with_dual_axis — Kernel / GEIR /
ACLNN / E2E.

Fused operator = SwiGLU activation (mode 0) + dual-axis (-1 / -2) dynamic MX
block quantization with FOUR outputs: y1 / mx_scale1 / y2 / mx_scale2.

Math core (torch, CPU fp32):
    swiglu : torch.nn.functional.silu body (silu(x) = x * sigmoid(x)); the
             intermediate is rounded back to the input precision (fp16/bf16)
             before quantization (kernel intermediate storage contract).
    mx     : per 32-block dynamic MX quantization on both axes. OCP scale
             floor(log2(amax)) - emax or cuBLAS scale (FP8, mantissa-aware
             round-up). E8M0 exponent bytes, pair packing, -2 axis interleave
             and the grouped compact layout (group_index is CUMSUM mode:
             [floor(M/64)+G, N, 2] with per-group pair gaps) are encoding /
             layout steps.

E2E note: torch_npu exposes no npu_swiglu_mx_quant_with_dual_axis binding, so
the framework-layer entry is the in-repo torch extension
(quant/swiglu_mx_quant_with_dual_axis/torch_extension), reached as
cann_ops_nn.swiglu_mx_quant_with_dual_axis. dst_type there uses the CANN codes
(35/36/40/41) rather than torch ScalarType enums.
"""

import math

import numpy as np

__golden__ = {
    "kernel": {
        "swiglu_mx_quant_with_dual_axis": "swiglu_mx_quant_with_dual_axis_golden"
    },
    "aclnn": {
        "aclnnSwigluMxQuantWithDualAxis": "aclnn_swiglu_mx_quant_with_dual_axis_golden"
    },
    "e2e": {
        "cann_ops_nn.swiglu_mx_quant_with_dual_axis": "swiglu_mx_quant_with_dual_axis_e2e_golden"
    },
}

__input__ = {
    "aclnn": {"aclnnSwigluMxQuantWithDualAxis": "dual_axis_group_index_input"},
    "e2e": {
        "cann_ops_nn.swiglu_mx_quant_with_dual_axis": "dual_axis_group_index_input"
    },
}

__spec__ = {
    "swiglu_mx_quant_with_dual_axis": "SwigluMxQuantWithDualAxisKernelSpec",
    "aclnnSwigluMxQuantWithDualAxis": "AclnnSwigluMxQuantWithDualAxisSpec",
    "cann_ops_nn.swiglu_mx_quant_with_dual_axis": "SwigluMxQuantWithDualAxisE2ESpec",
}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    27: "bfloat16",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

TORCH_DTYPE_ENUM_TO_DST_TYPE = {
    23: 35,
    24: 36,
    291: 35,
    292: 36,
    296: 40,
    297: 41,
}

BLOCK_SIZE = 32

# dst_type -> (emax, mant_bits, min_exp, max_norm)
_DST_TYPE_MAP = {
    40: (2, 1, 0, 6.0),
    41: (0, 2, 0, 1.75),
    36: (8, 3, -6, 448.0),
    35: (15, 2, -14, 57344.0),
}


def _numpy_mx_dtype(mx_ele_dtype):
    if mx_ele_dtype == "float4_e2m1":
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if mx_ele_dtype == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if mx_ele_dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if mx_ele_dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if mx_ele_dtype == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    raise ValueError(f"Unsupported mx dtype: {mx_ele_dtype}")


def _get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


# ---------------------------------------------------------------------------
# Torch math core
# ---------------------------------------------------------------------------
def _torch_round_mode(v, round_mode):
    import torch

    if round_mode in ("rint", "even"):
        return torch.round(v)
    if round_mode in ("round", "nearest"):
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


def _torch_swiglu_mode0(x_t, activate_left, in_dtype_name):
    """SwiGLU mode 0 on the last axis via torch.nn.functional.silu."""
    import torch
    import torch.nn.functional as F

    h = x_t.shape[-1] // 2
    a = x_t[..., :h]
    b = x_t[..., h:]
    if activate_left:
        res = F.silu(a) * b
    else:
        res = F.silu(b) * a
    # round the SwiGLU intermediate back to the input precision
    if in_dtype_name == "bfloat16":
        res = res.to(torch.bfloat16).to(torch.float32)
    elif in_dtype_name == "float16":
        res = res.to(torch.float16).to(torch.float32)
    return res


def _torch_share_exp(amax, dst_type, scale_alg):
    """OCP / cuBLAS shared exponent (torch), clipped to the E8M0 range."""
    import torch

    emax, _, _, max_norm = _DST_TYPE_MAP[dst_type]
    if scale_alg == 1 and dst_type in (35, 36):
        s = (amax / max_norm).to(torch.float32)
        bits = s.view(torch.int32)
        exponents = ((bits & 0x7F800000) >> 23).to(torch.int16)
        mantissas = bits & 0x007FFFFF
        round_up = ((exponents > 0) & (exponents < 254) & (mantissas > 0)) | (
            (exponents == 0) & (mantissas > 2**22)
        )
        exponents = torch.where(round_up, exponents + 1, exponents)
        share = (exponents - 127).to(torch.float32)
    else:
        safe = torch.where(amax > 0, amax, torch.ones_like(amax))
        share = torch.floor(torch.log2(safe)) - emax
    share = torch.where(amax == 0, torch.full_like(share, float("-inf")), share)
    share = torch.where(share > 127, torch.full_like(share, float("nan")), share)
    share = torch.clamp(share, min=-127)
    return share


def _torch_quantize_elements(values, share_exp, dst_type, round_mode):
    import torch

    _, mant_bits, min_exp, max_norm = _DST_TYPE_MAP[dst_type]
    ret = values / torch.pow(2.0, share_exp)
    private_exp = torch.floor(torch.log2(torch.abs(ret) + (ret == 0).to(ret.dtype)))
    private_exp = torch.clamp(private_exp, min=min_exp)
    ret = ret / torch.pow(2.0, private_exp) * float(2**mant_bits)
    ret = _torch_round_mode(ret, round_mode)
    ret = ret / float(2**mant_bits) * torch.pow(2.0, private_exp)
    return torch.clamp(ret, min=-max_norm, max=max_norm)


def _torch_axis_quant(data_t, axis, dst_type, block_size, round_mode, scale_alg):
    """Single-axis MX quantization (torch math + numpy E8M0 layout encode).

    Returns (y_t torch fp32, scale_np uint8 [pairs...]).
    """
    import torch

    ndim = data_t.dim()
    axis = axis % ndim
    shape = list(data_t.shape)
    pre_q = math.prod(shape[:axis]) if axis > 0 else 1
    q_dim = shape[axis]
    post_q = math.prod(shape[axis + 1 :]) if axis + 1 < len(shape) else 1
    flat = data_t.reshape(pre_q, q_dim, post_q)

    n_blocks = math.ceil(q_dim / block_size)
    n_pairs = (n_blocks + 1) // 2
    pad_len = n_blocks * block_size - q_dim
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, 0, 0, pad_len))
    blocks = flat.reshape(pre_q, n_blocks, block_size, post_q)

    amax = torch.amax(torch.abs(blocks), dim=2, keepdim=True)
    share_exp = _torch_share_exp(amax, dst_type, scale_alg)
    q = _torch_quantize_elements(blocks, share_exp, dst_type, round_mode)
    q = torch.nan_to_num(q, nan=0.0)  # NPU casts NaN elements to +0
    y_t = q.reshape(pre_q, n_blocks * block_size, post_q)[:, :q_dim, :].reshape(shape)

    # E8M0 bytes + pair packing (slot k of pair p = block 2p+k; pad byte 0)
    se_np = share_exp[:, :, 0, :].cpu().numpy().astype(np.float32)
    scale_bytes = np.where(
        np.isnan(se_np), np.uint8(255), np.clip(se_np + 127.0, 0, 255)
    ).astype(np.uint8)
    padded = np.zeros((pre_q, n_pairs * 2, post_q), dtype=np.uint8)
    padded[:, :n_blocks, :] = scale_bytes
    pairs = padded.reshape(pre_q, n_pairs, 2, post_q)
    lead = tuple(shape[:axis])
    trail = tuple(shape[axis + 1 :])
    tmp = pairs.reshape(lead + (n_pairs, 2) + trail)
    k_axis = len(lead) + 1
    perm = [i for i in range(tmp.ndim) if i != k_axis] + [k_axis]
    scale_out = np.ascontiguousarray(np.transpose(tmp, perm))
    return y_t, scale_out


def _torch_grouped_axis_rows(
    data_t, group_index, dst_type, block_size, round_mode, scale_alg
):
    """Grouped -2 axis quantization (group_index CUMSUM mode).

    Layout (mirrors the merged grouped golden / kernel infershape): work in
    the BLOCK-ROW domain — group g owns ceil(rows_g / 32) block rows padded
    to even with zero rows, inter-group gaps are filled with 1.0 (E8M0 byte
    127); the allocated pair rows = M // 64 + G. Rows are then folded into
    pairs [pair, col, slot].
    """
    import torch

    rows, n_cols = data_t.shape
    pair_rows = rows // (block_size * 2) + len(group_index)
    block_rows = pair_rows * 2
    block_scale = np.zeros((block_rows, n_cols), dtype=np.uint8)
    y_t = torch.zeros_like(data_t)

    cur_idx = 0
    prev = 0
    start = 0
    for idx, cum in enumerate(group_index):
        cum = int(cum)
        gv = cum - prev
        prev = cum
        # every group (including zero-row ones) advances the placement chain
        # (verified on Ascend950PR: a 0-row group still occupies its slot)
        next_group_start = (cum // (block_size * 2) + idx + 1) * 2
        if gv > 0 and start < rows:
            end = min(start + gv, rows)
            y_part, sp = _torch_axis_quant(
                data_t[start:end], 0, dst_type, block_size, round_mode, scale_alg
            )
            y_t[start:end] = y_part
            # pairs (npair, n_cols, 2) -> block rows (npair*2, n_cols)
            sp_blocks = sp.transpose(0, 2, 1).reshape(-1, n_cols)
            n_group_blocks = math.ceil(gv / block_size)
            real_idx = cur_idx + n_group_blocks
            pad_idx = math.ceil(real_idx / 2) * 2
            block_scale[cur_idx:real_idx] = sp_blocks[:n_group_blocks]
            # [real_idx:pad_idx] stays zero (2^-127); gap [pad_idx:next_group_start] = 1.0
            block_scale[pad_idx:next_group_start] = 127
            start = end
        cur_idx = next_group_start

    scale_np = (
        block_scale.reshape(pair_rows, 2, n_cols)
        .transpose(0, 2, 1)
        .reshape(pair_rows, n_cols, 2)
    )
    return y_t, scale_np


def _torch_dual_axis(x, group_index, kwargs):
    """Full chain: returns (y1_t, scale1_np, y2_t, scale2_np) with torch math
    and numpy E8M0 layout encoding."""
    import torch

    activate_left = bool(kwargs.get("activate_left", True))
    round_mode = str(kwargs.get("round_mode", "rint"))
    scale_alg = int(kwargs.get("scale_alg", 0))
    dst_type = int(kwargs.get("dst_type", 35))
    max_dtype_value = float(kwargs.get("max_dtype_value", 0.0))
    if max_dtype_value != 0.0:
        raise ValueError(
            f"max_dtype_value is reserved and only 0.0 is supported, got {max_dtype_value}"
        )

    if hasattr(x, "detach"):
        x_t = x.detach().cpu()
        in_name = str(x_t.dtype).replace("torch.", "")
        if x_t.dtype == torch.bfloat16:
            x_t = x_t.to(torch.float32)
        else:
            x_t = x_t.to(torch.float32)
    else:
        arr = np.asarray(x)
        in_name = str(arr.dtype)
        x_t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))

    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    if dst_type not in _DST_TYPE_MAP:
        raise NotImplementedError(f"Not support dst_type {dst_type} yet!")

    swiglu_t = _torch_swiglu_mode0(x_t, activate_left, in_name)

    # -1 axis quant (group-independent)
    y1_t, scale1_np = _torch_axis_quant(
        swiglu_t, -1, dst_type, BLOCK_SIZE, round_mode, scale_alg
    )
    # -2 axis quant: grouped (cumsum) when group_index is present
    if group_index is None:
        y2_t, scale2_np = _torch_axis_quant(
            swiglu_t, -2, dst_type, BLOCK_SIZE, round_mode, scale_alg
        )
    else:
        gi = np.asarray(group_index).reshape(-1).astype(np.int64)
        y2_t, scale2_np = _torch_grouped_axis_rows(
            swiglu_t, gi, dst_type, BLOCK_SIZE, round_mode, scale_alg
        )
    return y1_t, scale1_np, y2_t, scale2_np, _numpy_mx_dtype(dst_type_str)


def _pack_fp4_to_uint8(ele):
    if not isinstance(ele, np.ndarray) or "float4" not in str(ele.dtype):
        return ele
    if ele.shape[-1] % 2:
        raise ValueError("FP4 packing requires an even last dim")
    raw = ele.view(np.uint8) & 0x0F
    pairs = raw.reshape(raw.shape[:-1] + (-1, 2))
    packed = pairs[..., 0].astype(np.uint16) | (pairs[..., 1].astype(np.uint16) << 4)
    return packed.astype(np.uint8)


def _y_to_device_repr(y_np, dst_np_dtype, dst_type):
    y_q = y_np.astype(dst_np_dtype)
    if dst_type in (40, 41):
        return _pack_fp4_to_uint8(y_q)
    return y_q


# ---------------------------------------------------------------------------
# Kernel / GEIR golden
# ---------------------------------------------------------------------------
def swiglu_mx_quant_with_dual_axis_golden(
    x,
    group_index=None,
    *,
    activate_left=True,
    round_mode="rint",
    scale_alg=0,
    dst_type=35,
    max_dtype_value=0.0,
    **kwargs,
):
    """Golden for swiglu_mx_quant_with_dual_axis (kernel/GEIR path).

    Parameter names/order follow swiglu_mx_quant_with_dual_axis_def.cpp
    (inputs positional, attributes keyword). All input tensors numpy.ndarray.

    Returns: (y1, mx_scale1, y2, mx_scale2)
    """
    y1_t, s1_np, y2_t, s2_np, dst_dtype = _torch_dual_axis(
        x,
        group_index,
        dict(
            activate_left=activate_left,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type=dst_type,
            max_dtype_value=max_dtype_value,
        ),
    )
    y1 = y1_t.cpu().numpy().astype(dst_dtype)
    y2 = y2_t.cpu().numpy().astype(dst_dtype)
    from en_dtypes import float8_e8m0

    return [
        y1,
        s1_np.view(float8_e8m0),
        y2,
        s2_np.view(float8_e8m0),
    ]


# ---------------------------------------------------------------------------
# ACLNN golden (C header parameter order)
# ---------------------------------------------------------------------------
def aclnn_swiglu_mx_quant_with_dual_axis_golden(
    x,
    groupIndexOptional=None,
    activateLeft=True,
    roundModeOptional="rint",
    scaleAlg=0,
    dstType=35,
    maxDtypeValue=0.0,
    y1Out=None,
    mxscale1Out=None,
    y2Out=None,
    mxscale2Out=None,
    **kwargs,
):
    """Golden for aclnnSwigluMxQuantWithDualAxis (GetWorkspaceSize minus
    workspaceSize/executor)."""
    return swiglu_mx_quant_with_dual_axis_golden(
        x,
        groupIndexOptional,
        activate_left=bool(activateLeft),
        round_mode=str(roundModeOptional) if roundModeOptional is not None else "rint",
        scale_alg=int(scaleAlg),
        dst_type=int(dstType),
        max_dtype_value=float(maxDtypeValue),
    )


# ---------------------------------------------------------------------------
# E2E golden (torch extension signature; device representations)
# ---------------------------------------------------------------------------
def swiglu_mx_quant_with_dual_axis_e2e_golden(
    x,
    group_index=None,
    *,
    activate_left=True,
    round_mode="rint",
    scale_alg=0,
    dst_type=40,
    max_dtype_value=0.0,
    **kwargs,
):
    """E2E golden for cann_ops_nn.swiglu_mx_quant_with_dual_axis.

    dst_type accepts both the CANN codes used by the torch extension and the
    torch ScalarType enums. Returns the four outputs in the device-side
    representation: y1/y2 as fp8 native dtypes or FP4 packed uint8,
    mx_scale1/2 as uint8 bit-views of E8M0.
    """
    dst = TORCH_DTYPE_ENUM_TO_DST_TYPE.get(int(dst_type), int(dst_type))
    if dst not in _DST_TYPE_MAP:
        raise ValueError(f"Unsupported dst_type: {dst_type}")
    y1_t, s1_np, y2_t, s2_np, dst_dtype = _torch_dual_axis(
        x,
        group_index,
        dict(
            activate_left=activate_left,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type=dst,
            max_dtype_value=max_dtype_value,
        ),
    )
    y1 = _y_to_device_repr(y1_t.cpu().numpy(), dst_dtype, dst)
    y2 = _y_to_device_repr(y2_t.cpu().numpy(), dst_dtype, dst)
    return [
        np.ascontiguousarray(y1),
        np.ascontiguousarray(s1_np),
        np.ascontiguousarray(y2),
        np.ascontiguousarray(s2_np),
    ]


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------
def dual_axis_group_index_input(x, group_index=None, *extra, **kwargs):
    """Normalize the CUMSUM-mode group_index in place (non-decreasing, last
    element == rows)."""
    if group_index is None:
        return
    xshape = tuple(x.shape) if hasattr(x, "shape") else np.asarray(x).shape
    rows = int(xshape[0])
    if hasattr(group_index, "copy_"):
        import torch

        arr = torch.as_tensor(group_index).to(torch.int64)
        arr = torch.cumsum(torch.sort(arr.clamp(min=0)).values, 0)
        arr = torch.clamp(arr, max=rows)
        arr[-1] = rows
        group_index.copy_(arr)
    else:
        arr = np.asarray(group_index).reshape(-1).astype(np.int64)
        arr = np.sort(np.clip(arr, 0, None))
        arr = np.minimum(np.cumsum(arr), rows)
        arr[-1] = rows
        group_index[...] = arr


# ---------------------------------------------------------------------------
# Custom compare: four outputs, per-dtype standards
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


def _mx_element_compare(actual, expected, output_name):
    """FP8 -> 1-LSB (requant); FP4 packed / other -> binary."""
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
    lsb_rel = 2.0**-3 if "float8_e4m3" in dtype_name else 2.0**-2
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


def _grouped_valid_pair_rows(group_index, rows):
    """Pair rows covered by the CUMSUM-mode groups (compact placement)."""
    gi = np.asarray(group_index).reshape(-1).astype(np.int64)
    # device placement chain (verified on Ascend950PR): group idx works in the
    # BLOCK-ROW domain, next group start = (cum // 64 + idx + 1) * 2; every
    # group (including zero-row ones) advances the chain. Valid pair rows are
    # the pairs covered by each non-empty group's blocks.
    valid = []
    cur = 0
    prev = 0
    for idx, cum in enumerate(gi):
        cum = int(cum)
        gv = cum - prev
        prev = cum
        next_start = (cum // 64 + idx + 1) * 2
        if gv > 0:
            n_blocks = math.ceil(gv / 32)
            real_end = min(cur + n_blocks, next_start)
            valid.extend(range(cur // 2, (real_end + 1) // 2))
        cur = next_start
    return np.asarray(sorted(set(valid)), dtype=np.int64)


def swiglu_mx_quant_with_dual_axis_compare(
    y1_npu,
    mxscale1_npu,
    y2_npu,
    mxscale2_npu,
    y1_gold,
    mxscale1_gold,
    y2_gold,
    mxscale2_gold,
    compare_context=None,
    **kwargs,
):
    """Per-output compare: y1/y2 per dtype standard; mx_scale1 full binary
    (-1 axis, no group gaps); mx_scale2 binary on the rows covered by
    group_index (cumsum) or all rows when un-grouped."""
    gi = None
    if compare_context is not None:
        its = compare_context.input_tensors
        if its is not None and len(its) > 1 and its[1] is not None:
            gi = np.asarray(its[1]).reshape(-1).astype(np.int64)

    results = [
        _mx_element_compare(y1_npu, y1_gold, "y1"),
        _bin_compare(mxscale1_npu, mxscale1_gold, "mx_scale1"),
        _mx_element_compare(y2_npu, y2_gold, "y2"),
    ]

    a2 = _ensure_np(mxscale2_npu)
    e2 = _ensure_np(mxscale2_gold)
    if gi is not None and a2.ndim == 3:
        rows = a2.shape[1]
        valid = _grouped_valid_pair_rows(gi, rows)
        if valid.size and valid[-1] < a2.shape[0] and valid[-1] < e2.shape[0]:
            results.append(_bin_compare(a2[valid], e2[valid], "mx_scale2(valid rows)"))
        else:
            results.append(_bin_compare(a2, e2, "mx_scale2"))
    else:
        results.append(_bin_compare(a2, e2, "mx_scale2"))
    return results


# ---------------------------------------------------------------------------
# third_party: independent torch composition reference
# ---------------------------------------------------------------------------
class SwigluMxQuantWithDualAxisTorchRef:
    """Independent third-party reference: pure torch composition.

    torch.nn.functional.silu for the activation; MX block quantization with
    plain torch ops (amax/log2/floor/pow/round) on both axes; the E8M0 byte
    encoding is delegated to the layout encoder.
    """

    def __call__(
        self,
        x,
        group_index=None,
        *,
        activate_left=True,
        round_mode="rint",
        scale_alg=0,
        dst_type=35,
        max_dtype_value=0.0,
        **kwargs,
    ):
        import torch
        import torch.nn.functional as F

        dst = int(TORCH_DTYPE_ENUM_TO_DST_TYPE.get(dst_type, dst_type))
        emax, mant_bits, min_exp, max_norm = _DST_TYPE_MAP[dst]

        xf = x.detach().cpu()
        in_prec = str(xf.dtype).replace("torch.", "")
        xf = xf.float()
        h = xf.shape[-1] // 2
        a, b = xf[..., :h], xf[..., h:]
        y = (F.silu(a) * b) if activate_left else (F.silu(b) * a)
        if in_prec == "bfloat16":
            y = y.to(torch.bfloat16).float()
        elif in_prec == "float16":
            y = y.to(torch.float16).float()

        def quant_axis(t, axis):
            n = t.shape[axis]
            moved = t.movedim(axis, 0)
            nb = (n + 31) // 32
            pad = nb * 32 - n
            if pad:
                moved = F.pad(moved, (0, 0) * (moved.dim() - 1) + (0, pad))
            blocks = moved.reshape(nb, 32, *moved.shape[1:])
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
            pe = torch.floor(torch.log2(torch.abs(v) + (v == 0).float())).clamp(
                min=min_exp
            )
            v = v / torch.pow(2.0, pe) * float(2**mant_bits)
            if round_mode in ("round", "nearest"):
                neg = torch.signbit(v)
                v = torch.where(
                    neg,
                    -torch.floor(torch.abs(v) + 0.5),
                    torch.floor(torch.abs(v) + 0.5),
                )
            else:
                v = getattr(torch, round_mode if round_mode != "rint" else "round")(v)
            v = v / float(2**mant_bits) * torch.pow(2.0, pe)
            v = torch.clamp(v, min=-max_norm, max=max_norm)
            v = v.reshape(nb * 32, *moved.shape[1:])[:n]
            return v.movedim(0, axis), share

        y1, _ = quant_axis(y, -1)
        y2, _ = quant_axis(y, -2)
        dst_np = _numpy_mx_dtype(DATA_TYPE_INT_TO_STR[dst])
        return [y1.numpy().astype(dst_np), y2.numpy().astype(dst_np)]


# ---------------------------------------------------------------------------
# TestSpec classes
# ---------------------------------------------------------------------------
class SwigluMxQuantWithDualAxisKernelSpec:
    """Kernel/GEIR spec (numpy I/O)."""

    golden = swiglu_mx_quant_with_dual_axis_golden
    compare = swiglu_mx_quant_with_dual_axis_compare
    third_party = {"torch": SwigluMxQuantWithDualAxisTorchRef}
    tolerance = {
        "float8_e4m3fn": {"standard": "binary_equal"},
        "float8_e5m2": {"standard": "binary_equal"},
        "float4_e2m1": {"standard": "binary_equal"},
        "float4_e1m2": {"standard": "binary_equal"},
        "float8_e8m0": {"standard": "binary_equal"},
    }


class AclnnSwigluMxQuantWithDualAxisSpec:
    """ACLNN spec (C header parameter order)."""

    golden = aclnn_swiglu_mx_quant_with_dual_axis_golden
    compare = swiglu_mx_quant_with_dual_axis_compare
    third_party = {"torch": SwigluMxQuantWithDualAxisTorchRef}
    tolerance = SwigluMxQuantWithDualAxisKernelSpec.tolerance


class SwigluMxQuantWithDualAxisE2ESpec:
    """E2E spec for the in-repo torch extension entry (device representations)."""

    golden = swiglu_mx_quant_with_dual_axis_e2e_golden
    compare = swiglu_mx_quant_with_dual_axis_compare
    third_party = {"torch": SwigluMxQuantWithDualAxisTorchRef}
    tolerance = SwigluMxQuantWithDualAxisKernelSpec.tolerance
