# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""MXFP8/MXFP4 reference and TTK kernel / ACLNN / PyTorch TestSpecs."""

import math
from functools import lru_cache

import ml_dtypes
import numpy as np

try:  # TTK resolves float8_e8m0 through en_dtypes; mirror it so binary_equal
    # sees identical dtype strings on both sides (fallback keeps portability).
    from en_dtypes import float8_e8m0 as np_float8_e8m0
except ModuleNotFoundError:  # pragma: no cover
    np_float8_e8m0 = ml_dtypes.float8_e8m0fnu

try:
    from en_dtypes import float4_e2m1 as np_float4_e2m1
except ModuleNotFoundError:  # pragma: no cover
    np_float4_e2m1 = ml_dtypes.float4_e2m1fn


__spec__ = {
    "quant_matmul_activation_quant": "QuantMatmulActivationQuantTestSpec",
    "aclnnQuantMatmulActivationQuant": "QuantMatmulActivationQuantAclnnTestSpec",
    "aclnnQuantMatmulActivationQuantWeightNz": "QuantMatmulActivationQuantAclnnTestSpec",
    "cann_ops_nn.ops.quant_matmul_activation_quant": "QuantMatmulActivationQuantE2ETestSpec",
}


def fp8_dtype(dtype):
    """Normalize GE dtype codes and framework dtype names."""
    if dtype in (23, 35) or "e5m2" in str(dtype).lower():
        return "float8_e5m2"
    if dtype in (24, 36) or "e4m3" in str(dtype).lower():
        return "float8_e4m3fn"
    raise ValueError("The MXFP8 reference requires E4M3FN or E5M2")


def mx_dtype(dtype):
    """Normalize the supported MX element dtype names and enum values."""
    if dtype in (40, 296) or "float4_e2m1" in str(dtype).lower():
        return "float4_e2m1"
    return fp8_dtype(dtype)


_FP32_TINY = np.float32(np.finfo(np.float32).tiny)
# BF16 min normal (2**-126): a group whose amax falls below this is all-subnormal and
# gets a zero reciprocal in the OCP kernel lineage (zeroScaleOnZeroExp=true).
_MX_BF16_NORMAL_MIN = np.float64(2.0) ** -126
_QUANTIZE_CHUNK_ELEMENTS = 1 << 20


def _ftz_float32(value):
    """Flush FP32 subnormals to signed zero, as the 950 VF pipes do by default."""
    value = np.asarray(value, dtype=np.float32)
    signed_zero = np.copysign(np.float32(0.0), value)
    return np.where(np.abs(value) < _FP32_TINY, signed_zero, value).astype(
        np.float32, copy=False
    )


def _div_ftz_float32(lhs, rhs):
    """Model the kernel's FP32 Div under --cce-ftz=true (operands and quotient)."""
    lhs = _ftz_float32(lhs)
    rhs = _ftz_float32(rhs)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _ftz_float32(np.divide(lhs, rhs))


def decode_fp8(raw, dtype):
    """Decode all FP8 encodings, including signed zero, subnormals and NaNs."""
    raw = np.asarray(raw, dtype=np.uint8)
    is_e5m2 = fp8_dtype(dtype) == "float8_e5m2"
    mantissa_bits, bias = (2, 15) if is_e5m2 else (3, 7)
    exponent = (raw & 0x7F) >> mantissa_bits
    mantissa = raw & ((1 << mantissa_bits) - 1)
    fraction = mantissa.astype(np.float32) / (1 << mantissa_bits)
    normal = np.ldexp(1 + fraction, exponent.astype(np.int32) - bias)
    subnormal = np.ldexp(fraction, 1 - bias)
    result = np.where(exponent == 0, subnormal, normal)
    if is_e5m2:
        result = np.where(
            exponent == 31, np.where(mantissa == 0, np.inf, np.nan), result
        )
    else:
        result = np.where((exponent == 15) & (mantissa == 7), np.nan, result)
    return np.copysign(result, np.where(raw & 0x80, -1.0, 1.0)).astype(np.float32)


@lru_cache(maxsize=2)
def _fp8_positive_table(dtype):
    """Cache the finite nonnegative FP8 grid used by every quantization block."""
    dtype = fp8_dtype(dtype)
    max_code = 0x7B if dtype == "float8_e5m2" else 0x7E
    table = decode_fp8(np.arange(max_code + 1, dtype=np.uint8), dtype).astype(
        np.float64
    )
    table.setflags(write=False)
    return table


def encode_fp8(values, dtype):
    """Round-to-nearest-even mirroring the hardware FP32->FP8 SAT cast.

    The kernel uses a saturating FP32->FP8 cast. Therefore values beyond the
    largest finite FP8 value, including +/-inf, clamp to the corresponding
    signed finite maximum; NaN is converted to zero by the default FP8
    saturation mode. RNE is used for values inside the finite range.
    """
    dtype = fp8_dtype(dtype)
    values = np.asarray(values, dtype=np.float64)
    table = _fp8_positive_table(dtype)
    max_code = len(table) - 1
    magnitude = np.abs(values)
    # SAT clamps as soon as the input is outside the finite range. Do not use
    # the midpoint to the first non-finite encoding: that is the NO_SAT rule.
    overflow = magnitude > table[-1]
    in_range = np.where(overflow, table[-1], magnitude)
    upper = np.minimum(np.searchsorted(table, in_range), max_code)
    lower = np.maximum(upper - 1, 0)
    lower_distance = in_range - table[lower]
    upper_distance = table[upper] - in_range
    use_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    raw = np.where(use_upper, upper, lower).astype(np.uint8)
    raw = np.where(overflow, max_code, raw)
    raw = raw | np.where(np.signbit(values), 0x80, 0).astype(np.uint8)
    return np.where(np.isnan(values), np.uint8(0), raw).astype(np.uint8)


_FP4_E2M1_POSITIVE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64
)


def decode_fp4(raw):
    """Decode unpacked FLOAT4_E2M1 nibbles to FP32."""
    raw = np.asarray(raw, dtype=np.uint8) & np.uint8(0x0F)
    magnitude = _FP4_E2M1_POSITIVE[raw & np.uint8(0x07)]
    return np.copysign(magnitude, np.where(raw & 0x08, -1.0, 1.0)).astype(np.float32)


def encode_fp4(values, round_mode="rint"):
    """Cast FP32 values to unpacked FLOAT4_E2M1 nibbles."""
    if round_mode not in ("rint", "floor", "round"):
        raise ValueError("MXFP4 supports rint, floor or round")
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.minimum(np.abs(values), _FP4_E2M1_POSITIVE[-1])
    upper = np.minimum(
        np.searchsorted(_FP4_E2M1_POSITIVE, magnitude),
        len(_FP4_E2M1_POSITIVE) - 1,
    )
    lower = np.maximum(upper - 1, 0)
    exact = _FP4_E2M1_POSITIVE[upper] == magnitude
    if round_mode == "floor":
        index = np.where(np.signbit(values), upper, np.where(exact, upper, lower))
    else:
        lower_distance = magnitude - _FP4_E2M1_POSITIVE[lower]
        upper_distance = _FP4_E2M1_POSITIVE[upper] - magnitude
        if round_mode == "round":
            use_upper = upper_distance <= lower_distance
        else:
            use_upper = (upper_distance < lower_distance) | (
                (upper_distance == lower_distance) & ((upper & 1) == 0)
            )
        index = np.where(use_upper, upper, lower)
    raw = index.astype(np.uint8) | np.where(np.signbit(values), 0x08, 0).astype(
        np.uint8
    )
    return np.where(np.isnan(values), np.uint8(0), raw).astype(np.uint8)


def decode_e8m0(raw):
    """E8M0 byte 0 denotes 2**-127; byte 255 denotes NaN (there is no zero)."""
    raw = np.asarray(raw, dtype=np.uint8)
    result = np.ldexp(np.ones(raw.shape, dtype=np.float32), raw.astype(np.int32) - 127)
    return np.where(raw == 255, np.nan, result).astype(np.float32)


def _quantize_mx_fp4(values, scale_alg, round_mode, dst_type_max):
    """Quantize BF16-rounded values to MXFP4, matching the GELU epilogue."""
    if scale_alg not in (0, 2):
        raise ValueError("MXFP4 supports scale_alg 0 or 2")
    if round_mode not in ("rint", "floor", "round"):
        raise ValueError("MXFP4 supports rint, floor or round")
    if scale_alg == 2 and not (dst_type_max == 0.0 or 6.0 <= dst_type_max <= 12.0):
        raise ValueError("MXFP4 scale_alg 2 requires dst_type_max=0 or [6, 12]")

    values = np.asarray(values, dtype=np.float32)
    width = values.shape[-1]
    rows = values.reshape(-1, width)
    quantized = np.empty(rows.shape, dtype=np.uint8)
    scales = np.zeros((len(rows), (width + 63) // 64 * 2), dtype=np.uint8)
    block_count = (width + 31) // 32
    padded_width = block_count * 32
    rows_per_chunk = max(1, _QUANTIZE_CHUNK_ELEMENTS // padded_width)

    for row_start in range(0, len(rows), rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, len(rows))
        row_chunk = rows[row_start:row_end]
        chunk_rows = row_end - row_start
        if padded_width == width:
            padded = row_chunk
        else:
            padded = np.zeros((chunk_rows, padded_width), dtype=np.float32)
            padded[:, :width] = row_chunk
        blocks = padded.reshape(chunk_rows, block_count, 32)
        finite = np.isfinite(blocks).all(axis=-1)
        amax = np.max(np.abs(blocks), axis=-1).astype(np.float32)

        zero_reciprocal = np.zeros_like(amax, dtype=bool)
        if scale_alg == 0:
            _, exponent = np.frexp(amax)
            scale_codes = np.clip(exponent - 1 - 2 + 127, 0, 254).astype(np.uint8)
            scale_codes[amax == 0.0] = 0
            # OCP reduces BF16 exponent fields. An all-zero/subnormal group has
            # maxExp==0 and the GELU path explicitly writes a zero reciprocal.
            max_exp_bits = amax.view(np.uint32) & np.uint32(0x7F800000)
            zero_reciprocal = max_exp_bits == 0
        else:
            effective_max = 6.0 if dst_type_max == 0.0 else dst_type_max
            if effective_max in (6.0, 7.0):
                # The dedicated dynamic-range path uses a zero reciprocal when
                # sharedExp is zero. OCP/cuBLAS instead use 2**127 here, so an
                # E8M0 byte of zero must not generally force the data to zero.
                bf16_bits = (amax.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
                exp_bits = bf16_bits & np.uint16(0x7F80)
                add_bits = np.uint16(0x001F if effective_max == 7.0 else 0x003F)
                rounded_exp = (
                    bf16_bits.astype(np.uint32) + np.uint32(add_bits)
                ) & np.uint32(0x7F80)
                rounded_exp = np.where(
                    exp_bits < np.uint16(0x0100), np.uint32(0x0100), rounded_exp
                )
                scale_codes = ((rounded_exp - np.uint32(0x0100)) >> 7).astype(np.uint8)
                zero_reciprocal = scale_codes == 0
            else:
                scaled = amax * np.float32(1.0 / effective_max)
                bits = scaled.view(np.uint32)
                exponent = (bits & np.uint32(0x7F800000)) >> np.uint32(23)
                mantissa = bits & np.uint32(0x007FFFFF)
                increment = ((exponent > 0) & (exponent < 254) & (mantissa > 0)) | (
                    (exponent == 0) & (mantissa > (1 << 22))
                )
                scale_codes = (exponent + increment.astype(np.uint32)).astype(np.uint8)
                scale_codes[amax == 0.0] = 0

        scale_codes[~finite] = 255
        scales[row_start:row_end, :block_count] = scale_codes
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = np.ldexp(
                blocks.astype(np.float64),
                (127 - scale_codes.astype(np.int32))[..., None],
            )
        normalized = np.where(zero_reciprocal[..., None], 0.0, normalized)
        encoded = encode_fp4(normalized, round_mode)
        encoded[~finite] = np.uint8(0)
        quantized[row_start:row_end] = encoded.reshape(chunk_rows, padded_width)[
            :, :width
        ]

    return (
        quantized.reshape(values.shape),
        scales.reshape(values.shape[:-1] + ((width + 63) // 64, 2)),
    )


def quantize_mx(values, output_dtype, scale_alg=0, round_mode="rint", dst_type_max=0.0):
    """Quantize FP32 values in 32-element groups, stored as ceil(N/64),2."""
    output_dtype = mx_dtype(output_dtype)
    if output_dtype == "float4_e2m1":
        return _quantize_mx_fp4(values, scale_alg, round_mode, dst_type_max)
    if scale_alg not in (0, 1):
        raise ValueError("MXFP8 supports scale_alg 0 or 1")
    if round_mode != "rint":
        raise ValueError("MXFP8 supports round_mode rint")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ValueError("Quantization requires a nonempty last dimension")
    width = values.shape[-1]
    rows = values.reshape(-1, width)
    quantized = np.empty(rows.shape, dtype=np.uint8)
    scales = np.zeros((len(rows), (width + 63) // 64 * 2), dtype=np.uint8)
    emax = 15 if output_dtype == "float8_e5m2" else 8
    block_count = (width + 31) // 32
    padded_width = block_count * 32
    rows_per_chunk = max(1, _QUANTIZE_CHUNK_ELEMENTS // padded_width)

    for row_start in range(0, len(rows), rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, len(rows))
        row_chunk = rows[row_start:row_end]
        chunk_rows = row_end - row_start
        if padded_width == width:
            padded = row_chunk
        else:
            padded = np.zeros((chunk_rows, padded_width), dtype=np.float32)
            padded[:, :width] = row_chunk
        blocks = padded.reshape(chunk_rows, block_count, 32)

        finite = np.isfinite(blocks).all(axis=-1)
        amax = np.max(np.abs(blocks), axis=-1).astype(np.float64)
        amax[~finite] = 0.0
        # A finite group whose members are all BF16 subnormals or zeros has a zero raw
        # BF16 exponent field. The OCP kernel lineage (zeroScaleOnZeroExp=true, used by
        # the swiglu/gelu_mx epilogues) writes a zero reciprocal for such groups, so the
        # whole group quantizes to zero. cuBLAS (scale_alg=1) keeps the computed scale.
        subnormal_group = np.logical_and(finite, amax < _MX_BF16_NORMAL_MIN)
        if scale_alg == 0:
            zero_reciprocal = subnormal_group
        else:
            zero_reciprocal = np.zeros_like(subnormal_group)
        mantissa, exponent = np.frexp(amax)
        exponent -= 1 + emax
        # qmax is exactly 1.75 * 2**emax for both FP8 formats.
        if scale_alg == 1:
            exponent += mantissa * 2 > 1.75
        codes = np.clip(exponent + 127, 0, 254).astype(np.uint8)
        codes[amax == 0.0] = 0
        codes[~finite] = 255
        scales[row_start:row_end, :block_count] = codes

        normalized = np.ldexp(
            blocks.astype(np.float64),
            (127 - codes.astype(np.int32))[..., None],
        )
        normalized = np.where(zero_reciprocal[..., None], 0.0, normalized)
        encoded = encode_fp8(normalized, output_dtype)
        encoded[~finite] = np.uint8(0x7F)
        quantized[row_start:row_end] = encoded.reshape(chunk_rows, padded_width)[
            :, :width
        ]
    return (
        quantized.reshape(values.shape),
        scales.reshape(values.shape[:-1] + ((width + 63) // 64, 2)),
    )


def _dequantize_operand(values, scales, k_axis):
    # Finite FP8 * E8M0 can exceed FP32 even when the scaled dot product is finite.
    # Keep reference operands wide until the complete matmul result is rounded to FP32.
    values = np.asarray(values, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    k = values.shape[k_axis]
    groups = (k + 63) // 64
    expected = values.shape[:-2] + (
        (values.shape[-2], groups, 2) if k_axis == -1 else (groups, values.shape[-1], 2)
    )
    if scales.shape != expected:
        raise ValueError(f"MX scale shape must be {expected}, got {scales.shape}")
    if k_axis == -1:
        expanded = np.repeat(scales.reshape(scales.shape[:-2] + (-1,)), 32, axis=-1)[
            ..., :k
        ]
    else:
        paired = np.swapaxes(scales, -1, -2)
        expanded = np.repeat(
            paired.reshape(paired.shape[:-3] + (-1, values.shape[-1])), 32, axis=-2
        )
        expanded = expanded[..., :k, :]
    return values * expanded


def matmul_mx(
    x1, x2, x1_scale, x2_scale, bias=None, transpose_x1=False, transpose_x2=False
):
    """Broadcast all batch axes; add the full-N FP32 bias before activation."""
    if x1_scale is None or x2_scale is None:
        raise ValueError("MX inputs require explicit x1_scale and x2_scale")
    x1, x2 = np.asarray(x1), np.asarray(x2)
    if not 2 <= x1.ndim <= 6 or not 2 <= x2.ndim <= 6 or min(*x1.shape, *x2.shape) <= 0:
        raise ValueError("Inputs must be nonempty tensors with rank 2 through 6")
    a = _dequantize_operand(x1, x1_scale, -2 if transpose_x1 else -1)
    b = _dequantize_operand(x2, x2_scale, -1 if transpose_x2 else -2)
    if transpose_x1:
        a = np.swapaxes(a, -1, -2)
    if transpose_x2:
        b = np.swapaxes(b, -1, -2)
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("The contraction dimensions must match")
    result = np.matmul(a, b).astype(np.float32)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        valid_shapes = [(result.shape[-1],)]
        if result.ndim == 3:
            valid_shapes.append((result.shape[0], 1, result.shape[-1]))
        if bias.shape not in valid_shapes:
            raise ValueError("Bias must be [N], or [B,1,N] for a rank-3 output")
        result = result + bias
    return result


def activate(values, activation_type):
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        if activation_type == "swiglu":
            if values.shape[-1] <= 0 or values.shape[-1] % 64:
                raise ValueError(
                    "SwiGLU requires positive pre-activation N divisible by 64"
                )
            gate, linear = np.split(values, 2, axis=-1)
            # Mirror the kernel FP32 chain (Muls -> Exp -> Adds -> Div -> Mul)
            # under --cce-ftz=true: Exp and Div flush subnormal inputs/outputs,
            # while the final Mul and the BF16 cast keep them (the subnormal_bias
            # kernel UT relies on a subnormal SiLU(gate)*linear surviving into gluRes).
            exp_neg = _ftz_float32(np.exp(_ftz_float32(-gate)))
            denom = np.float32(1.0) + exp_neg
            silu_gate = _div_ftz_float32(gate, denom)
            return silu_gate * linear
        if activation_type == "gelu_tanh":
            return (
                0.5
                * values
                * (
                    1
                    + np.tanh(
                        np.float32(math.sqrt(2 / math.pi))
                        * (values + np.float32(0.044715) * values**3)
                    )
                )
            )
        if activation_type == "gelu_erf":
            erf = np.vectorize(math.erf, otypes=[np.float32])(
                values / np.float32(math.sqrt(2))
            )
            return 0.5 * values * (1 + erf)
    raise ValueError("Unsupported activation_type")


def round_bfloat16(values):
    """RNE BF16 boundary used by the activation epilogues, in an FP32 array."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    rounded = np.where(np.isnan(values), bits | np.uint32(0x00400000), rounded).astype(
        np.uint32
    )
    return rounded.view(np.float32)


def reference(
    x1,
    x2,
    bias,
    x1_scale,
    x2_scale,
    *,
    activation_type="gelu_tanh",
    output_dtype="float8_e4m3fn",
    scale_alg=0,
    round_mode="rint",
    dst_type_max=0.0,
    transpose_x1=False,
    transpose_x2=False,
):
    if activation_type == "swiglu" and mx_dtype(output_dtype) == "float4_e2m1":
        raise ValueError("SwiGLU requires MXFP8")
    matrix = matmul_mx(x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2)
    activated = round_bfloat16(activate(matrix, activation_type))
    return quantize_mx(
        activated,
        output_dtype,
        scale_alg,
        round_mode=round_mode,
        dst_type_max=dst_type_max,
    )


def _numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    return value.detach().cpu().float().numpy()


def _is_fractal_nz(formats, index=1):
    if formats is None or len(formats) <= index:
        return False
    return "FRACTAL_NZ" in str(formats[index]).upper()


def _nz_storage_shape(logical_shape, dtype):
    """Return the physical FRACTAL_NZ shape for an FP8/FP4 logical view."""
    logical_shape = tuple(int(dim) for dim in logical_shape)
    if len(logical_shape) < 2 or min(logical_shape) <= 0:
        raise ValueError("WeightNZ requires a nonempty logical matrix")
    m, n = logical_shape[-2:]
    c0 = 64 if mx_dtype(dtype) == "float4_e2m1" else 32
    return logical_shape[:-2] + ((n + c0 - 1) // c0, (m + 15) // 16, 16, c0)


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def _numpy_storage_view(value, physical_shape):
    """Recover the physical storage behind a possibly logical NumPy view."""
    owner = value
    current = value
    visited = set()
    while getattr(current, "base", None) is not None and id(current) not in visited:
        visited.add(id(current))
        current = current.base
        if isinstance(current, np.ndarray):
            owner = current

    itemsize = value.dtype.itemsize
    value_addr = value.__array_interface__["data"][0]
    owner_addr = owner.__array_interface__["data"][0]
    byte_offset = value_addr - owner_addr
    if byte_offset < 0 or byte_offset % itemsize:
        raise ValueError("Invalid WeightNZ NumPy storage offset")
    offset = byte_offset // itemsize
    flat = np.asarray(owner).reshape(-1)
    count = math.prod(physical_shape)
    if offset + count > flat.size:
        raise ValueError(
            f"WeightNZ storage has {flat.size - offset} elements, expected at least {count}"
        )
    return flat[offset : offset + count].reshape(physical_shape)


def _physical_nz_to_numpy(value, physical_shape):
    # Kernel ST passes the physical tensor itself.  TTK's ND->NZ transform is
    # a non-contiguous NumPy transpose, so use its indexed values instead of
    # walking to the pre-transpose base array.  ACLNN passes a logical view and
    # therefore takes the backing-storage path below.
    if tuple(value.shape) == tuple(physical_shape):
        return _numpy(value)
    if isinstance(value, np.ndarray):
        return _numpy(_numpy_storage_view(value, physical_shape))

    count = math.prod(physical_shape)
    offset = value.storage_offset()
    storage_size = value.untyped_storage().nbytes() // value.element_size()
    if offset + count > storage_size:
        raise ValueError(
            f"WeightNZ storage has {storage_size - offset} elements, expected at least {count}"
        )
    physical = value.as_strided(
        physical_shape, _contiguous_strides(physical_shape), storage_offset=offset
    )
    return _numpy(physical)


def weight_nz_to_nd(value, logical_shape, dtype=None):
    """Decode FP8/FP4 FRACTAL_NZ storage to its original logical ND matrix.

    The same mapping covers both x2 transpose modes.  For transposeX2=false
    logical_shape ends in (K, N); for transposeX2=true it ends in (N, K).
    The existing matmul path applies transposeX2 after this layout conversion.
    """
    logical_shape = tuple(int(dim) for dim in logical_shape)
    physical_shape = _nz_storage_shape(
        logical_shape, value.dtype if dtype is None else dtype
    )
    physical = _physical_nz_to_numpy(value, physical_shape)
    batch_rank = len(physical_shape) - 4
    axes = tuple(range(batch_rank)) + (
        batch_rank + 1,
        batch_rank + 2,
        batch_rank,
        batch_rank + 3,
    )
    padded = np.transpose(physical, axes).reshape(
        logical_shape[:-2]
        + (
            physical_shape[-3] * physical_shape[-2],
            physical_shape[-4] * physical_shape[-1],
        )
    )
    return padded[..., : logical_shape[-2], : logical_shape[-1]]


def _pack_fp4(raw):
    """Pack consecutive unpacked FP4 nibbles into uint8 bytes."""
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.shape[-1] % 2:
        raise ValueError("Packed MXFP4 requires an even last dimension")
    pairs = raw.reshape(raw.shape[:-1] + (-1, 2)).astype(np.uint16)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)


def _unpack_fp4(value):
    """Decode the packed uint8 representation used by the PyTorch API."""
    if "float4_e2m1" in str(value.dtype).lower():
        return _numpy(value)
    if isinstance(value, np.ndarray):
        packed = np.asarray(value, dtype=np.uint8)
    else:
        packed = value.detach().cpu().numpy().astype(np.uint8, copy=False)
    nibbles = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    nibbles[..., 0::2] = packed & np.uint8(0x0F)
    nibbles[..., 1::2] = packed >> np.uint8(4)
    return decode_fp4(nibbles)


def _raw_outputs(raw_y, raw_scale, dtype, torch_output=False, pack_fp4_output=False):
    """Return quantized y codes and E8M0 scale bytes in storage dtypes.

    TTK's default comparison then routes each output independently and reports
    one precision per output: FP8 y through mix_tolerance on decoded values,
    FP4 y and E8M0 y_scale through binary_equal on their raw codes.
    """
    dtype = mx_dtype(dtype)
    raw_y = np.ascontiguousarray(raw_y, dtype=np.uint8)
    raw_scale = np.ascontiguousarray(raw_scale, dtype=np.uint8)
    if dtype == "float4_e2m1":
        if pack_fp4_output:
            packed_y = np.ascontiguousarray(_pack_fp4(raw_y))
            if torch_output:
                import torch

                return [
                    torch.from_numpy(packed_y),
                    torch.from_numpy(raw_scale).view(torch.float8_e8m0fnu),
                ]
            return [packed_y, raw_scale.view(np_float8_e8m0)]
        # TTK expands 4-bit kernel/ACLNN outputs before comparison. Keep the
        # native NumPy dtype here even when its input carrier was a torch tensor.
        return [raw_y.view(np_float4_e2m1), raw_scale.view(np_float8_e8m0)]

    is_e5m2 = dtype == "float8_e5m2"
    if torch_output:
        import torch

        return [
            torch.from_numpy(raw_y).view(
                torch.float8_e5m2 if is_e5m2 else torch.float8_e4m3fn
            ),
            torch.from_numpy(raw_scale).view(torch.float8_e8m0fnu),
        ]
    return [
        raw_y.view(ml_dtypes.float8_e5m2 if is_e5m2 else ml_dtypes.float8_e4m3fn),
        raw_scale.view(np_float8_e8m0),
    ]


# Per-output precision standard for TTK's default comparison routing
# (Spec.tolerance). y routes to mix_tolerance with the plain double-permille
# standard: an element is off when its relative error exceeds 1 permille
# (rtol=0.001, atol=0), and at most 1 permille of the elements may be off
# (required_matched_ratio=0.999). For FP8 outputs an adjacent-code difference
# already has a 3%-12% relative error, so the off-elements are exactly the
# code mismatches and the standard reads as a bit-error rate. The plain
# standard sets no outlier-magnitude cap, so max_abs_error_limit is the
# unreachable decoded-value spread (2x the finite max: E4M3 896, E5M2
# 114688); float('inf') is not usable because the metrics must survive the
# CSV literal_eval round trip. NaN/Inf mismatches still fail unconditionally
# (TTK built-in). FP4 y and y_scale route to binary_equal: every output code
# and scale byte must match exactly, so a wrong scale algorithm cannot pass by
# compensating the quantized values.
_MX_TOLERANCE = {
    "float4_e2m1": {"standard": "binary_equal"},
    "float4_e2m1fn": {"standard": "binary_equal"},
    "float8_e4m3fn": {
        "rtol": 0.001,
        "atol": 0.0,
        "required_matched_ratio": 0.999,
        "max_abs_error_limit": 896.0,
    },
    "float8_e5m2": {
        "rtol": 0.001,
        "atol": 0.0,
        "required_matched_ratio": 0.999,
        "max_abs_error_limit": 114688.0,
    },
}


class QuantMatmulActivationQuantTestSpec:
    tolerance = _MX_TOLERANCE

    @staticmethod
    def golden(
        x1,
        x2,
        bias,
        x1_scale,
        x2_scale,
        *,
        activation_type="gelu_tanh",
        y_dtype=36,
        transpose_x1=False,
        transpose_x2=False,
        scale_alg=0,
        quant_mode="mx",
        round_mode="rint",
        dst_type_max=0.0,
        **kwargs,
    ):
        dtype = mx_dtype(y_dtype)
        if quant_mode != "mx":
            raise ValueError("This TestSpec covers MX quantization")
        input_formats = kwargs.get("input_formats", ())
        if _is_fractal_nz(input_formats):
            if activation_type == "swiglu" and transpose_x1:
                raise ValueError("SwiGLU WeightNZ requires transpose_x1=false")
            input_ori_shapes = kwargs.get("input_ori_shapes", ())
            if len(input_ori_shapes) <= 1 or input_ori_shapes[1] is None:
                raise ValueError("WeightNZ kernel golden requires input_ori_shapes[1]")
            x2 = weight_nz_to_nd(x2, input_ori_shapes[1], dtype)
        # TTK kernel path normalizes the first output dtype to float32, so the
        # MX kind must come from the y_dtype attribute (GE code 35/36/40 or name).
        result = reference(
            *map(_numpy, (x1, x2, bias, x1_scale, x2_scale)),
            activation_type=activation_type,
            output_dtype=dtype,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type_max=dst_type_max,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
        )
        return _raw_outputs(*result, dtype)


class QuantMatmulActivationQuantAclnnTestSpec:
    tolerance = _MX_TOLERANCE

    @staticmethod
    def golden(
        x1,
        x2,
        x1ScaleOptional,
        x2Scale,
        biasOptional=None,
        transposeX1=False,
        transposeX2=False,
        groupSize=4295032864,
        activationType="gelu_tanh",
        quantMode="mx",
        roundMode="rint",
        scaleAlg=0,
        dstTypeMax=0.0,
        yOut=None,
        yScaleOut=None,
        **kwargs,
    ):
        del yScaleOut
        if groupSize not in (0, 4295032864) or quantMode != "mx":
            raise ValueError("Invalid MX attributes")
        # TTK aclnn passes a float32 placeholder as yOut and promotes tensor
        # dtypes, so the MX kind of y must come from the explicit yDtype
        # attribute (GE code 35/36/40 or dtype name) when present.
        dtype = kwargs.get("yDtype")
        if dtype is None:
            # yOut is a promoted float32 placeholder; TTK's output_dtypes (the
            # CSV tensor_dtypes) carries the real y kind and wins over yOut.
            output_dtypes = kwargs.get("output_dtypes")
            if output_dtypes:
                dtype = output_dtypes[0]
        if dtype is None:
            dtype = str(yOut.dtype) if yOut is not None else str(x1.dtype)
        dtype = mx_dtype(dtype)
        if _is_fractal_nz(kwargs.get("tensor_formats", ())):
            if activationType == "swiglu" and transposeX1:
                raise ValueError("SwiGLU WeightNZ requires transpose_x1=false")
            # TTK passes a logical x2 view backed by the physical NZ storage.
            # Recover and decode that storage before _numpy() copies the view.
            x2 = weight_nz_to_nd(x2, tuple(x2.shape), dtype)
        result = reference(
            *map(_numpy, (x1, x2, biasOptional, x1ScaleOptional, x2Scale)),
            activation_type=activationType,
            output_dtype=dtype,
            scale_alg=scaleAlg,
            round_mode=roundMode,
            dst_type_max=dstTypeMax,
            transpose_x1=transposeX1,
            transpose_x2=transposeX2,
        )
        return _raw_outputs(*result, dtype, torch_output=not isinstance(x1, np.ndarray))


class QuantMatmulActivationQuantE2ETestSpec:
    tolerance = _MX_TOLERANCE

    @staticmethod
    def golden(
        x1,
        x2,
        x2_scale,
        *,
        x1_scale=None,
        bias=None,
        output_dtype=None,
        activation_type="gelu_tanh",
        scale_alg=0,
        round_mode="rint",
        dst_type_max=0.0,
        x1_dtype=None,
        x2_dtype=None,
        **kwargs,
    ):
        # PyTorch views already express the logical matrix: no second transpose.
        dtype = output_dtype
        if dtype is None:
            dtype = x1_dtype if x1_dtype is not None else str(x1.dtype)
        dtype = mx_dtype(dtype)
        is_fp4 = dtype == "float4_e2m1"
        if is_fp4:
            if mx_dtype(x1_dtype) != dtype or mx_dtype(x2_dtype) != dtype:
                raise ValueError("MXFP4 E2E inputs require FP4 dtype attributes")
            x1_value = _unpack_fp4(x1)
            x2_value = _unpack_fp4(x2)
        else:
            x1_value = _numpy(x1)
            x2_value = _numpy(x2)
        result = reference(
            x1_value,
            x2_value,
            *map(_numpy, (bias, x1_scale, x2_scale)),
            activation_type=activation_type,
            output_dtype=dtype,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type_max=dst_type_max,
        )
        return _raw_outputs(
            *result,
            dtype,
            torch_output=not isinstance(x1, np.ndarray),
            pack_fp4_output=is_fp4,
        )
