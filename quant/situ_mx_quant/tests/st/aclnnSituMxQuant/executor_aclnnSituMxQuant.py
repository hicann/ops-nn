#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ATK executor for aclnnSituMxQuant (sample structure: BaseApi + AclnnBaseApi).

CPU golden is embedded below (the same numpy math that was bit-level cross-checked
against the kernel contract and verified on Ascend 950PR):
    situ   : beta*tanh(gate/beta)*sigmoid(gate); up=linearBeta*tanh(up/linearBeta) when linearBeta>0
    store  : Situ result is round-tripped through the input dtype (fp16/bf16) buffer
    mx     : per-32-element OCP MX; emax 36->8 / 35->15; E8M0 bytes
             (0x7f clamp, 0x00 for all-zero block, 0xff + NaN block for inf/nan)
    layout : y = x.shape[:-1] + [H]; yScale = x.shape[:-1] + [ceil(H/64), 2]
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

# ---------------------------------------------------------------------------
# CPU golden implementation (embedded; numpy bit-level kernel emulation)
# ---------------------------------------------------------------------------

try:  # CI installs ml-dtypes; the offline self-check uses the uint8 fallback.
    from ml_dtypes import bfloat16 as _np_bfloat16
except ImportError:  # pragma: no cover - fallback path
    _np_bfloat16 = None

try:
    from ml_dtypes import float8_e4m3fn as _np_f8_e4m3
    from ml_dtypes import float8_e5m2 as _np_f8_e5m2
except ImportError:  # pragma: no cover - fallback path
    _np_f8_e4m3 = None
    _np_f8_e5m2 = None

try:
    from en_dtypes import float4_e1m2 as _np_f4_e1m2
    from en_dtypes import float4_e2m1 as _np_f4_e2m1
    from en_dtypes import float8_e8m0 as _np_e8m0
except ImportError:  # pragma: no cover - fallback path
    _np_f4_e1m2 = None
    _np_f4_e2m1 = None
    _np_e8m0 = None


# ---------------------------------------------------------------------------
# Constants (mirror op_kernel/arch35/situ_mx_quant_common.h)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 32
SCALE_PAIR = 2  # y_scale trailing dim: two 32-element sub-blocks per 64-element group
FP32_EXP_BIAS = 127
BF16_EXP_MASK = 0x7F80
BF16_INF_FIELD = 0x7F80
FP8_NAN_CODE = 0x7F
FP4_NAN_NIBBLE = 0x0F

DST_EMAX = {36: 8, 35: 15, 40: 2, 41: 0}
DST_MAX_NORM = {36: 448.0, 35: 57344.0, 40: 6.0, 41: 1.75}
DST_NAMES = {
    36: "float8_e4m3fn",
    35: "float8_e5m2",
    40: "float4_e2m1",
    41: "float4_e1m2",
}
NAME_TO_DST = {v: k for k, v in DST_NAMES.items()}
FP8_DST = (36, 35)
FP4_DST = (40, 41)

# FP8 bit layouts: (mantissa bits, exponent bias, min normal exponent, max code)
FP8_INFO = {
    36: {
        "mant_bits": 3,
        "exp_bias": 7,
        "min_exp": -6,
        "max_norm": 448.0,
        "max_code": 0x7E,
    },
    35: {
        "mant_bits": 2,
        "exp_bias": 15,
        "min_exp": -14,
        "max_norm": 57344.0,
        "max_code": 0x7B,
    },
}

# E2M1 magnitude codes, 0..7 -> (0, .5, 1, 1.5, 2, 3, 4, 6); no NaN/Inf in E2M1.
E2M1_MAGS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E1M2_STEP = np.float32(0.25)
E1M2_MAX_CODE = 7


# ---------------------------------------------------------------------------
# Small numeric helpers (pure numpy bit ops)
# ---------------------------------------------------------------------------
def _round_mode_scalar(sv, round_mode):
    """Round a scaled (dimensionless) value to an integer grid."""
    if round_mode in ("rint", "even"):
        return np.rint(sv)
    if round_mode in ("round", "nearest"):
        return np.copysign(np.floor(np.abs(sv) + np.float32(0.5)), sv)
    if round_mode == "floor":
        return np.floor(sv)
    raise ValueError("Unrecognized round_mode: %s" % (round_mode,))


def _round_to_grid(values, step, round_mode, max_norm):
    """Round ``values`` onto a per-element ``step`` grid, then saturate."""
    v = np.asarray(values, dtype=np.float32)
    step = np.asarray(step, dtype=np.float32)
    q = _round_mode_scalar((v / step).astype(np.float32), round_mode)
    return np.clip((q * step).astype(np.float32), -max_norm, max_norm)


def _f32_to_bf16_bits(values):
    """Round-to-nearest-even fp32 -> bf16 bits (CAST_RINT), pure bit ops."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def _store_to_input_precision(values, tname):
    """``CAST<T, float, CAST_RINT / NO_SAT>`` then read back as fp32 (exact)."""
    v = np.asarray(values, dtype=np.float32)
    if tname == "float16":
        with np.errstate(over="ignore", invalid="ignore"):
            return v.astype(np.float16).astype(np.float32)
    return _bf16_bits_to_f32(_f32_to_bf16_bits(v))


def _input_exponent_fields(values_t_f32, tname):
    """bf16 exponent field (bits 14..7) of the value stored in the input dtype T.

    fp16 inputs go through ``CAST_HALF_TO_BF16`` in the kernel; the exponent is
    preserved for normals and fp16 sub-normals become bf16 normals, which is
    exactly the fp32 exponent of the (exact) fp16 value.  bf16 inputs keep their
    raw field (so bf16 sub-normals stay 0 == "zero block").
    """
    u = np.ascontiguousarray(values_t_f32, dtype=np.float32).view(np.uint32)
    if tname == "float16":
        return (((u >> np.uint32(23)) & np.uint32(0xFF)) << np.uint32(7)).astype(
            np.uint16
        )
    return ((u >> np.uint32(16)) & np.uint32(BF16_EXP_MASK)).astype(np.uint16)


# ---------------------------------------------------------------------------
# FP8 encode / decode (saturating RINT cast, bit exact)
# ---------------------------------------------------------------------------
def _fp8_encode_bytes(values, dst):
    """FP32 -> FP8 bytes: SatMode::SAT + RoundMode::CAST_RINT (common.h:297-300)."""
    info = FP8_INFO[dst]
    mb = info["mant_bits"]
    v = np.asarray(values, dtype=np.float32)
    nan_mask = np.isnan(v)
    sign_mask = np.signbit(v)
    mag = np.minimum(np.abs(v), np.float32(info["max_norm"]))
    mag = np.where(nan_mask, np.float32(0.0), mag)  # NaN handled at the end
    positive = mag > 0.0
    safe = np.where(positive, mag, np.float32(1.0))
    expo = np.floor(np.log2(safe)).astype(np.int32)
    expo = np.where(positive, expo, info["min_exp"])
    expo = np.maximum(expo, info["min_exp"])
    step = np.exp2((expo - mb).astype(np.float32))
    q = np.rint((mag / step).astype(np.float32)).astype(np.int64)
    # mantissa overflow 2^(mb+1) -> next exponent (e.g. 1.999 -> 2.0)
    carry = q >= (1 << (mb + 1))
    expo = np.where(carry, expo + 1, expo)
    q = np.where(carry, q >> 1, q)
    # exponent field 0 encodes the sub-normal mantissa directly
    sub = np.logical_and(expo <= info["min_exp"], q < (1 << mb))
    code = np.empty(q.shape, dtype=np.int64)
    code[sub] = q[sub]
    normal = np.logical_not(sub)
    code[normal] = ((expo[normal] + info["exp_bias"]) << mb) | (q[normal] - (1 << mb))
    code = np.minimum(code, info["max_code"])
    code = np.where(sign_mask, code | 0x80, code)
    code = np.where(nan_mask, FP8_NAN_CODE, code)
    return code.astype(np.uint8)


_FP8_TABLES = {}


def _fp8_decode_table(dst):
    table = _FP8_TABLES.get(dst)
    if table is not None:
        return table
    info = FP8_INFO[dst]
    mb = info["mant_bits"]
    bias = info["exp_bias"]
    out = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if (code & 0x80) else 1.0
        body = code & 0x7F
        exp_field = body >> mb
        mant = body & ((1 << mb) - 1)
        if exp_field == 0:
            value = mant * (2.0 ** (info["min_exp"] - mb))
        elif exp_field == (0x7F >> mb):
            if dst == 36 and body == 0x7F:
                out[code] = np.nan
                continue
            if dst == 35:
                out[code] = sign * (np.inf if mant == 0 else np.nan)
                continue
            value = (1.0 + mant / float(1 << mb)) * (2.0 ** (exp_field - bias))
        else:
            value = (1.0 + mant / float(1 << mb)) * (2.0 ** (exp_field - bias))
        out[code] = sign * value
    _FP8_TABLES[dst] = out
    return out


def _decode_fp_to_f32(arr, dst):
    """Device/golden element representation -> fp32 values (NaN/Inf preserved)."""
    a = np.asarray(arr)
    if a.dtype.kind == "f" and a.dtype.itemsize >= 2:
        return a.astype(np.float32)
    raw = np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.uint8)
    if dst in FP8_DST:
        return _fp8_decode_table(dst)[raw].reshape(a.shape)
    if dst == 40:
        mags = E2M1_MAGS[raw & 0x07]
        return np.where((raw & 0x08) != 0, -mags, mags).reshape(a.shape)
    mag = (raw & 0x07).astype(np.float32) * E1M2_STEP
    return np.where((raw & 0x08) != 0, -mag, mag).reshape(a.shape)


# ---------------------------------------------------------------------------
# FP4 encode (grid rounding, bit exact; FP4 has no Inf/NaN in E2M1)
# ---------------------------------------------------------------------------
def _fp4_encode_nibbles(values, dst, round_mode):
    """FP32 -> FP4 nibbles (low nibble per element, sign in bit 3)."""
    v = np.asarray(values, dtype=np.float32)
    nan_mask = np.isnan(v)
    if dst == 40:
        # E2M1: step = 2^(max(floor(log2|v|), 0) - 1), i.e. 0.5 / 1 / 2 / ...
        mag = np.abs(v)
        mag = np.where(nan_mask, np.float32(0.0), mag)
        positive = mag > 0.0
        safe = np.where(positive, mag, np.float32(1.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            expo = np.floor(np.log2(safe)).astype(np.float32)
        step = np.exp2(np.maximum(expo, np.float32(0.0)) - np.float32(1.0))
        grid = _round_to_grid(v, step, round_mode, 6.0)
        gmag = np.abs(grid)
        gsign = np.signbit(grid) | (grid == 0.0) & np.signbit(v)
        codes = np.abs(E2M1_MAGS.reshape(1, -1) - gmag.reshape(-1, 1)).argmin(axis=1)
        codes = codes.reshape(v.shape)
    else:
        # E1M2: uniform 0.25 grid, magnitude code 0..7 (max 1.75)
        grid = _round_to_grid(v, E1M2_STEP, round_mode, 1.75)
        gsign = np.signbit(grid) | (grid == 0.0) & np.signbit(v)
        codes = np.rint((np.abs(grid) / E1M2_STEP).astype(np.float32)).astype(np.int64)
        codes = np.clip(codes, 0, E1M2_MAX_CODE)
    nib = codes.astype(np.uint8)
    nib = np.where(gsign, nib | 0x08, nib)
    nib = np.where(nan_mask, FP4_NAN_NIBBLE, nib)
    return nib.reshape(v.shape)


# ---------------------------------------------------------------------------
# Materialisation (native dtype when the packages are available)
# ---------------------------------------------------------------------------
def _numpy_dtype_for(name):
    text = str(name or "").lower()
    if "float8_e4m3fn" in text:
        return _np_f8_e4m3
    if "float8_e5m2" in text and "fnuz" not in text:
        return _np_f8_e5m2
    if "e8m0" in text:
        return _np_e8m0
    if "float4_e2m1" in text:
        return _np_f4_e2m1
    if "float4_e1m2" in text:
        return _np_f4_e1m2
    return None


def _view_output(raw_u8, name):
    """Bit-reinterpret raw bytes as the requested output dtype (uint8 fallback)."""
    arr = np.ascontiguousarray(raw_u8, dtype=np.uint8)
    dtype = _numpy_dtype_for(name)
    if dtype is None:
        return arr
    try:
        return arr.view(dtype)
    except (TypeError, ValueError):  # pragma: no cover - exotic dtype layouts
        return arr


# ---------------------------------------------------------------------------
# Situ activation + MX quantisation core
# ---------------------------------------------------------------------------
def _resolve_input_dtype_name(x, kwargs):
    names = kwargs.get("input_dtypes") or ()
    candidates = [str(getattr(x, "dtype", ""))]
    if len(names) > 0 and names[0] is not None:
        candidates.append(str(names[0]))
    for candidate in candidates:
        text = candidate.lower()
        # NB: "bfloat16" contains "float16" as a substring -- check it first.
        if "bfloat16" in text or text.endswith("bf16"):
            return "bfloat16"
        if "float16" in text or text.endswith("fp16"):
            return "float16"
    if np.asarray(x).dtype == np.float16:
        return "float16"
    raise ValueError(
        "situ_mx_quant golden: cannot resolve the input dtype from %r/%r"
        % (getattr(x, "dtype", None), names[:1])
    )


def _x_to_float32(x):
    if hasattr(x, "detach"):  # torch tensor from the ACLNN/E2E path
        import torch

        return x.detach().cpu().to(torch.float32).numpy()
    arr = np.asarray(x)
    if arr.dtype == np.uint16:  # bf16 raw bits fallback (no ml_dtypes installed)
        return _bf16_bits_to_f32(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _situ_float32(x_f32, beta, linear_beta, activate_left):
    """Situ activation in fp32, mirroring ``ComputeVfSitu`` op order."""
    beta32 = np.float32(beta)
    inv_beta = np.float32(1.0) / beta32
    half = x_f32.shape[-1] // 2
    if activate_left:
        gate, up = x_f32[..., :half], x_f32[..., half:]
    else:
        gate, up = x_f32[..., half:], x_f32[..., :half]
    gate_div_beta = (gate * inv_beta).astype(np.float32)
    tanh_gate = np.tanh(gate_div_beta).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        sigmoid_gate = (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate))).astype(
            np.float32
        )
    situ_a = ((tanh_gate * sigmoid_gate).astype(np.float32) * beta32).astype(np.float32)
    if linear_beta > 0.0:
        lb32 = np.float32(linear_beta)
        inv_lb = np.float32(1.0) / lb32
        up = (
            np.tanh((up * inv_lb).astype(np.float32)).astype(np.float32) * lb32
        ).astype(np.float32)
    return (situ_a * up.astype(np.float32)).astype(np.float32)


def _mx_quantize(situ_stored, tname, dst, round_mode):
    """s_T (fp32, exact T values) -> (y element codes, y_scale bytes)."""
    s = np.asarray(situ_stored, dtype=np.float32)
    lead = s.shape[:-1]
    hidden = s.shape[-1]
    rows = int(np.prod(lead)) if len(lead) > 0 else 1
    flat = s.reshape(rows, hidden)

    n_blocks = (hidden + BLOCK_SIZE - 1) // BLOCK_SIZE
    pad = n_blocks * BLOCK_SIZE - hidden
    flat_pad = np.pad(flat, ((0, 0), (0, pad))) if pad else flat
    blocks = flat_pad.reshape(rows, n_blocks, BLOCK_SIZE)

    fields = _input_exponent_fields(flat_pad, tname).reshape(rows, n_blocks, BLOCK_SIZE)
    block_field = fields.max(axis=2).astype(np.int32)
    poison = block_field == BF16_INF_FIELD
    zero = block_field == 0
    # E8M0 scale exponent: e = E - emax, negative exponents are legal (byte = 127 + e).
    expo = ((block_field >> 7) - FP32_EXP_BIAS - DST_EMAX[dst]).astype(np.int32)
    expo = np.where(poison | zero, 0, expo)

    scale_byte = np.where(
        zero,
        np.int32(0x00),
        np.where(poison, np.int32(0xFF), np.clip(127 + expo, 0, 254)),
    ).astype(np.uint8)

    scaled = np.ldexp(blocks, (-expo)[..., None]).astype(np.float32)
    if dst in FP4_DST:
        y_codes = _fp4_encode_nibbles(scaled, dst, round_mode)
        nan_code = np.uint8(FP4_NAN_NIBBLE)
    else:
        y_codes = _fp8_encode_bytes(scaled, dst)
        nan_code = np.uint8(FP8_NAN_CODE)
    if poison.any():
        y_codes = np.where(poison[..., None], nan_code, y_codes)
    y_codes = y_codes.reshape(rows, n_blocks * BLOCK_SIZE)[:, :hidden]

    n_pairs = (n_blocks + 1) // 2
    padded = np.zeros((rows, n_pairs * SCALE_PAIR), dtype=np.uint8)
    padded[:, :n_blocks] = scale_byte
    scale_bytes = padded.reshape(tuple(lead) + (n_pairs, SCALE_PAIR))
    return y_codes.reshape(tuple(lead) + (hidden,)), scale_bytes


# ---------------------------------------------------------------------------
# Kernel golden (numpy in, numpy out; positional args = op_def inputs only)
# ---------------------------------------------------------------------------
def situ_mx_quant_golden(
    x,
    *,
    beta=1.0,
    linear_beta=0.0,
    activate_left=False,
    axis=-1,
    dst_type=40,
    round_mode="rint",
    **kwargs,
):
    """Golden for the ``situ_mx_quant`` kernel (numpy I/O).

    Parameters follow ``op_host/situ_mx_quant_def.cpp``: the only tensor input is
    ``x``; all six attributes are keyword arguments (defaults as in def.cpp).
    ``kwargs`` carries ``{input,output}_{dtypes,ori_shapes,formats,ori_formats}``,
    ``full_soc_version``, ``short_soc_version`` and ``testcase_name``.

    Returns
    -------
    [y, y_scale]
        ``y``     : ``lead + [H]`` quantised elements (float8/float4 dtype when
                    ml_dtypes/en_dtypes are installed, raw uint8 otherwise).
        ``y_scale``: ``lead + [ceil(H/64), 2]`` E8M0 bytes, padding slot included.
    """
    dst = int(dst_type)
    if int(axis) != -1:
        raise ValueError("situ_mx_quant only supports axis=-1, got %r" % (axis,))
    if dst not in DST_EMAX:
        raise ValueError(
            "situ_mx_quant dst_type must be one of {35, 36, 40, 41}, got %r" % (dst,)
        )
    mode = str(round_mode or "rint")
    if dst in FP8_DST and mode != "rint":
        raise ValueError("FP8 output (dst_type %d) requires round_mode='rint'" % (dst,))
    if mode not in ("rint", "round", "floor"):
        raise ValueError("Unrecognized round_mode: %r" % (mode,))

    out_names = list(kwargs.get("output_dtypes") or ())
    y_name = (
        str(out_names[0])
        if len(out_names) > 0 and out_names[0] is not None
        else DST_NAMES[dst]
    )
    scale_name = (
        str(out_names[1])
        if len(out_names) > 1 and out_names[1] is not None
        else "float8_e8m0"
    )
    if DST_NAMES.get(dst) is not None and "float" in y_name.lower():
        expected_dst = NAME_TO_DST.get(y_name.lower())
        if expected_dst is not None and expected_dst != dst:
            raise ValueError(
                "output_dtypes y=%r does not match dst_type=%d" % (y_name, dst)
            )

    tname = _resolve_input_dtype_name(x, kwargs)
    x_f32 = _x_to_float32(x)
    if x_f32.ndim < 1 or x_f32.shape[-1] % 2 != 0:
        raise ValueError("x last dim must be even, got shape %r" % (x_f32.shape,))

    situ_out = _situ_float32(
        x_f32, float(beta), float(linear_beta), bool(activate_left)
    )
    # the kernel round-trips the Situ result through the input dtype buffer
    situ_stored = _store_to_input_precision(situ_out, tname)
    y_codes, scale_bytes = _mx_quantize(situ_stored, tname, dst, mode)
    return [_view_output(y_codes, y_name), _view_output(scale_bytes, scale_name)]


_DST_Y_NAME = {35: "float8_e5m2", 36: "float8_e4m3fn"}
_TORCH_Y_DTYPE = {35: torch.float8_e5m2, 36: torch.float8_e4m3fn}


def _input_dtype_name(x):
    text = str(getattr(x, "dtype", "")).lower()
    if "bfloat16" in text:
        return "bfloat16"
    if "float16" in text:
        return "float16"
    raise ValueError(
        "situ_mx_quant ATK: x dtype must be float16/bfloat16, got %r" % (x.dtype,)
    )


def _to_torch_outputs(y, scale, dst_type):
    y_np = np.ascontiguousarray(y)
    raw = y_np.view(np.uint8) if y_np.dtype != np.uint8 else y_np
    y_t = torch.from_numpy(raw.copy()).view(_TORCH_Y_DTYPE[int(dst_type)])
    s_u8 = np.ascontiguousarray(np.asarray(scale).view(np.uint8)).copy()
    # E8M0 scale compared as raw bytes on both nodes (e8m0 has no numeric ops in ATK)
    scale_t = torch.from_numpy(s_u8)
    return y_t, scale_t


@register("ascend_method_situ_mx_quant")
class MethodSituMxQuant(BaseApi):
    """CPU golden + NPU (cann_ops_nn) executor, mirrors the quant sample layout."""

    def golden(
        self,
        x,
        beta=1.0,
        linear_beta=0.0,
        activate_left=False,
        axis=-1,
        dst_type=36,
        round_mode="rint",
        **kwargs,
    ):
        out_dtypes = (_DST_Y_NAME[int(dst_type)], "float8_e8m0")
        y, scale = situ_mx_quant_golden(
            x,
            beta=float(beta),
            linear_beta=float(linear_beta),
            activate_left=bool(activate_left),
            axis=int(axis),
            dst_type=int(dst_type),
            round_mode=str(round_mode),
            input_dtypes=(_input_dtype_name(x),),
            output_dtypes=out_dtypes,
        )
        return _to_torch_outputs(y, scale, int(dst_type))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kw = input_data.kwargs
        x = kw["x"]
        args = dict(
            beta=kw.get("beta", 1.0),
            linear_beta=kw.get("linearBeta", 0.0),
            activate_left=kw.get("activateLeft", False),
            axis=kw.get("axis", -1),
            dst_type=kw.get("dstType", 36),
            round_mode=kw.get("roundModeOptional", "rint"),
        )
        if self.device in ("cpu", "gpu"):
            return self.golden(x, **args)
        import cann_ops_nn

        torch.npu.synchronize()
        y, scale = cann_ops_nn.situ_mx_quant(
            x.npu(),
            beta=float(args["beta"]),
            linear_beta=float(args["linear_beta"]),
            activate_left=bool(args["activate_left"]),
            dst_type=int(args["dst_type"]),
            round_mode=str(args["round_mode"]),
        )
        torch.npu.synchronize()
        # compare E8M0 scale as raw bytes (same representation as the CPU golden)
        return y, scale.view(torch.uint8)


@register("aclnn_situ_mx_quant")
class AclnnSituMxQuantApi(AclnnBaseApi):
    """Direct aclnn C-API executor (PyAclnn), mirrors the quant sample layout.

    aclnnSituMxQuantGetWorkspaceSize(x, beta, linearBeta, activateLeft, axis,
    dstType, roundModeOptional, yOut, yScaleOut, ...): no optional tensors.
    """

    def after_call(self, output_packages):
        outputs = [self.acl_tensor_to_torch(p) for p in output_packages]
        return outputs[0], outputs[1]
