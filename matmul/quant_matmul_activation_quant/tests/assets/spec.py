# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""MXFP8/MXFP4 reference and TTK kernel / ACLNN / PyTorch TestSpecs."""

import hashlib
import math
from functools import lru_cache

import ml_dtypes
import numpy as np

try:  # Match TTK's native storage dtype, with an ml_dtypes fallback.
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
    if dtype in (23, 35, 291) or "e5m2" in str(dtype).lower():
        return "float8_e5m2"
    if dtype in (24, 36, 292) or "e4m3" in str(dtype).lower():
        return "float8_e4m3fn"
    raise ValueError("The MXFP8 reference requires E4M3FN or E5M2")


def mx_dtype(dtype):
    """Normalize the supported MX element dtype names and enum values."""
    if dtype in (40, 296) or "float4_e2m1" in str(dtype).lower():
        return "float4_e2m1"
    return fp8_dtype(dtype)


def _mx_operand_dtypes(
    x1, x2, output_dtype, input_dtypes=(), x1_dtype=None, x2_dtype=None
):
    """Resolve logical MX types before promoted inputs lose their native dtype."""
    input_dtypes = input_dtypes or ()
    declared = (x1_dtype, x2_dtype)
    operands = []
    for index, value in enumerate((x1, x2)):
        if value is None:
            raise ValueError("MX inputs require x1 and x2")
        dtype = declared[index]
        if dtype is None:
            dtype = input_dtypes[index] if index < len(input_dtypes) else value.dtype
            if str(dtype).removeprefix("torch.") in ("float32", "float64"):
                dtype = str(value.dtype).removeprefix("torch.")
                # ACLNN may promote both the values and their dtype metadata.
                if dtype in ("float32", "float64"):
                    dtype = output_dtype
        operands.append(mx_dtype(dtype))
    if operands[0] != output_dtype:
        raise ValueError("MX output dtype must match x1")
    if (operands[0] == "float4_e2m1") != (operands[1] == "float4_e2m1"):
        raise ValueError("MXFP4 requires both inputs to be FP4")
    return tuple(operands)


_FP32_TINY = np.float32(np.finfo(np.float32).tiny)
_QUANTIZE_CHUNK_ELEMENTS = 1 << 20
_MX_GROUP_SIZE = 32


def _ftz_float32(value):
    """Flush FP32 subnormals to signed zero, as the dav_3510 VF pipes do by default."""
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
    values = np.asarray(values)
    if values.dtype == np.float32:
        # Native casts implement RNE; clip first to reproduce the SAT overflow
        # behavior and explicitly preserve the kernel's NaN -> zero rule.
        is_e5m2 = dtype == "float8_e5m2"
        native_dtype = ml_dtypes.float8_e5m2 if is_e5m2 else ml_dtypes.float8_e4m3fn
        maximum = np.float32(57344.0 if is_e5m2 else 448.0)
        raw = np.clip(values, -maximum, maximum).astype(native_dtype).view(np.uint8)
        return np.where(np.isnan(values), np.uint8(0), raw)
    # Keep FP64 inputs wide: narrowing values next to a midpoint can change RNE.
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


@lru_cache(maxsize=2)
def _fp4_rounding_boundaries(round_mode):
    boundaries = (_FP4_E2M1_POSITIVE[:-1] + _FP4_E2M1_POSITIVE[1:]) * 0.5
    if round_mode == "rint":
        # searchsorted(side="left") keeps ties below a boundary. Move odd
        # lower-code boundaries down one FP64 step so those ties round up.
        boundaries[1::2] = np.nextafter(boundaries[1::2], -np.inf)
    boundaries.setflags(write=False)
    return boundaries


def decode_fp4(raw):
    """Decode unpacked FLOAT4_E2M1 nibbles to FP32."""
    raw = np.asarray(raw, dtype=np.uint8) & np.uint8(0x0F)
    magnitude = _FP4_E2M1_POSITIVE[raw & np.uint8(0x07)]
    return np.copysign(magnitude, np.where(raw & 0x08, -1.0, 1.0)).astype(np.float32)


def encode_fp4(values, round_mode="rint"):
    """Cast FP32 values to unpacked FLOAT4_E2M1 nibbles."""
    if round_mode not in ("rint", "floor", "round"):
        raise ValueError("MXFP4 supports rint, floor or round")
    values = np.asarray(values)
    if values.dtype == np.float32 and round_mode == "rint":
        raw = values.astype(ml_dtypes.float4_e2m1fn).view(np.uint8)
        return np.where(np.isnan(values), np.uint8(0), raw)
    values = np.asarray(values, dtype=np.float64)
    negative = np.signbit(values)
    if round_mode == "floor":
        magnitude = np.minimum(np.abs(values), _FP4_E2M1_POSITIVE[-1])
        upper = np.minimum(
            np.searchsorted(_FP4_E2M1_POSITIVE, magnitude),
            len(_FP4_E2M1_POSITIVE) - 1,
        )
        index = upper - (~negative & (_FP4_E2M1_POSITIVE[upper] != magnitude))
    else:
        index = np.searchsorted(
            _fp4_rounding_boundaries(round_mode),
            np.abs(values),
            side="left" if round_mode == "rint" else "right",
        )
    raw = index.astype(np.uint8) | (negative.astype(np.uint8) << np.uint8(3))
    return np.where(np.isnan(values), np.uint8(0), raw).astype(np.uint8)


def decode_e8m0(raw):
    """E8M0 byte 0 denotes 2**-127; byte 255 denotes NaN (there is no zero)."""
    raw = np.asarray(raw, dtype=np.uint8)
    result = np.ldexp(np.ones(raw.shape, dtype=np.float32), raw.astype(np.int32) - 127)
    return np.where(raw == 255, np.nan, result).astype(np.float32)


def _quantize_mx_fp4(values, scale_alg, round_mode, dst_type_max):
    """Quantize BF16-rounded values to MXFP4, matching the GELU epilogue."""
    # ACLNN/Torch accepts a double, but the tiling attribute and kernel use FP32.
    dst_type_max = float(np.float32(dst_type_max))
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
            # Match DynamicMxQuant: sharedExp==0 also clears tiny normal groups.
            zero_reciprocal = scale_codes == 0
        else:
            effective_max = 6.0 if dst_type_max == 0.0 else dst_type_max
            if effective_max in (6.0, 7.0):
                # The dedicated dynamic-range path uses a zero reciprocal when
                # sharedExp is zero, as OCP does. cuBLAS can use 2**127 here, so an
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
        # QuantizeFp4Vf multiplies the BF16 activation and reciprocal into a
        # BF16 register before the FP4 cast. Preserve that rounding boundary:
        # a tiny negative product may round to -0 here, whereas a direct FP64
        # -> FP4 floor cast would incorrectly produce -0.5.
        normalized = round_bfloat16(normalized)
        encoded = encode_fp4(normalized, round_mode)
        encoded[~finite] = np.uint8(0)
        quantized[row_start:row_end] = encoded.reshape(chunk_rows, padded_width)[
            :, :width
        ]

    return (
        quantized.reshape(values.shape),
        scales.reshape(values.shape[:-1] + ((width + 63) // 64, 2)),
    )


def quantize_mx(
    values,
    output_dtype,
    scale_alg=0,
    round_mode="rint",
    dst_type_max=0.0,
):
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
        mantissa, exponent = np.frexp(amax)
        exponent -= 1 + emax
        # qmax is exactly 1.75 * 2**emax for both FP8 formats.
        if scale_alg == 1:
            exponent += mantissa * 2 > 1.75
        codes = np.clip(exponent + 127, 0, 254).astype(np.uint8)
        codes[amax == 0.0] = 0
        codes[~finite] = 255
        scales[row_start:row_end, :block_count] = codes

        # Scaling FP32 by a power of two is exact in the FP8 rounding range.
        # Any FP32 overflow saturates at the FP8 cast; FP32 underflow is already
        # below the smallest FP8 rounding boundary. This enables native casts.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = np.ldexp(blocks, (127 - codes.astype(np.int32))[..., None])
        if scale_alg == 0:
            # All activation epilogues follow DynamicMxQuant's sharedExp == 0 rule.
            normalized = np.where(
                codes[..., None] == 0, np.copysign(np.float32(0.0), blocks), normalized
            )
        # GenScale emits a NaN reciprocal for any group containing Inf/NaN.
        # E8M0 code 0xFF is NaN, not the finite exponent used by ldexp above:
        # all members of that group become NaN before the FP8 SAT cast, which
        # converts NaN to zero under the kernel's default CTRL[50] setting.
        normalized[~finite] = np.nan
        encoded = encode_fp8(normalized, output_dtype)
        quantized[row_start:row_end] = encoded.reshape(chunk_rows, padded_width)[
            :, :width
        ]
    return (
        quantized.reshape(values.shape),
        scales.reshape(values.shape[:-1] + ((width + 63) // 64, 2)),
    )


def _dequantize_operand(values, scales, k_axis):
    # Finite FP8 * E8M0 can exceed FP32 even when the scaled dot product is finite.
    # Keep reference operands wide; round the accumulator at each MX group below.
    values = np.asarray(values)
    scales = np.asarray(scales, dtype=np.float64)
    k = values.shape[k_axis]
    groups = (k + 63) // 64
    expected = values.shape[:-2] + (
        (values.shape[-2], groups, 2) if k_axis == -1 else (groups, values.shape[-1], 2)
    )
    if scales.shape != expected:
        raise ValueError(f"MX scale shape must be {expected}, got {scales.shape}")
    # Broadcast one scale per group without materializing a K-wide repeat or
    # an FP64 input copy. Keep N contiguous for the right-hand operand.
    result = np.empty(values.shape, dtype=np.float64)
    full_groups, tail = divmod(k, _MX_GROUP_SIZE)
    full = full_groups * _MX_GROUP_SIZE
    if k_axis == -1:
        scales = scales.reshape(scales.shape[:-2] + (-1,))
        if full_groups:
            shape = values.shape[:-1] + (full_groups, _MX_GROUP_SIZE)
            np.multiply(
                values[..., :full].reshape(shape),
                scales[..., :full_groups, None],
                out=result[..., :full].reshape(shape),
                dtype=np.float64,
            )
        if tail:
            np.multiply(
                values[..., full:],
                scales[..., full_groups, None],
                out=result[..., full:],
                dtype=np.float64,
            )
    else:
        scales = np.swapaxes(scales, -1, -2).reshape(
            values.shape[:-2] + (-1, values.shape[-1])
        )
        if full_groups:
            shape = values.shape[:-2] + (full_groups, _MX_GROUP_SIZE, values.shape[-1])
            np.multiply(
                values[..., :full, :].reshape(shape),
                scales[..., :full_groups, None, :],
                out=result[..., :full, :].reshape(shape),
                dtype=np.float64,
            )
        if tail:
            np.multiply(
                values[..., full:, :],
                scales[..., full_groups, None, :],
                out=result[..., full:, :],
                dtype=np.float64,
            )
    return result


def matmul_mx(
    x1, x2, x1_scale, x2_scale, bias=None, transpose_x1=False, transpose_x2=False
):
    """Accumulate MX K groups in FP32, initializing the first group with bias.

    FP64 group dot products avoid premature operand dequantization overflow.
    The running accumulator is rounded to FP32 after each 32-element MX group;
    bias participates from the first group instead of being added after MatMul.
    This models the accumulation boundary, not Cube's internal reduction tree.
    """
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
    output_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (
        a.shape[-2],
        b.shape[-1],
    )
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        valid_shapes = [(output_shape[-1],)]
        if len(output_shape) == 3:
            valid_shapes.append((output_shape[0], 1, output_shape[-1]))
        if bias.shape not in valid_shapes:
            raise ValueError("Bias must be [N], or [B,1,N] for a rank-3 output")
        result = np.broadcast_to(bias, output_shape).copy()
    else:
        result = np.zeros(output_shape, dtype=np.float32)
    partial = np.empty(output_shape, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, a.shape[-1], _MX_GROUP_SIZE):
            end = min(start + _MX_GROUP_SIZE, a.shape[-1])
            np.matmul(a[..., start:end], b[..., start:end, :], out=partial)
            partial += result
            np.copyto(result, partial, casting="unsafe")
    return result


def _fma_float32(lhs, rhs, addend):
    """Model one FP32 fused multiply-add, rather than two FP32 roundings."""
    return (
        np.asarray(lhs, dtype=np.float64) * np.asarray(rhs, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64)
    ).astype(np.float32)


def _erf_kernel_float32(values):
    """Model arch35 gelu.h::ErfCallee for implementation-consistency testing.

    Preserve the coefficient bit patterns, FP32 Horner FMA boundaries, and
    exp-based large-argument branch. This is the kernel's approximation, not
    a higher-accuracy replacement for math.erf.
    """
    values = np.asarray(values, dtype=np.float32)
    split = np.array(0x3F8060FE, dtype=np.uint32).view(np.float32)
    p_large = np.array(
        [
            0x38EB4C3A,
            0xBAAE005B,
            0x3C09919F,
            0xBD24D99A,
            0x3E235519,
            0x3F69B4F9,
            0x3F210A14,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    p_small = np.array(
        [
            0x38B1E96A,
            0xBA574D20,
            0x3BAAD5EA,
            0xBCDC1BE7,
            0x3DE718AF,
            0xBEC093AC,
            0x3E0375D3,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    magnitude = np.abs(values)
    large = magnitude >= split
    argument = np.where(large, magnitude, values * values)
    polynomial = np.where(large, p_large[0], p_small[0])
    for large_coeff, small_coeff in zip(p_large[1:], p_small[1:]):
        polynomial = _fma_float32(
            polynomial, argument, np.where(large, large_coeff, small_coeff)
        )
    tail = np.where(large, -magnitude, values)
    polynomial = _fma_float32(polynomial, tail, tail)
    exponent = polynomial * np.float32(math.log(2.0))
    exp_value = _ftz_float32(np.exp(_ftz_float32(exponent)))
    large_result = np.copysign(np.float32(1.0) - exp_value, values)
    # Use the kernel's final LT selection so NaN follows its exp branch too.
    return np.where(magnitude < split, polynomial, large_result)


def activate(values, activation_type, *, kernel_erf=True):
    """Apply activation; kernel_erf=False retains the math.erf diagnostic path."""
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
        if activation_type == "silu":
            # Match standalone SiLU: x / (1 + exp(-x)) with FP32 FTZ division.
            exp_neg = _ftz_float32(np.exp(_ftz_float32(-values)))
            denom = np.float32(1.0) + exp_neg
            return _div_ftz_float32(values, denom)
        if activation_type == "gelu_tanh":
            # Match GeluV2's constants and Mul -> Mul -> Axpy -> Muls order.
            tanh_approx_factor = np.float32(1.0 / 0.044715)
            neg_sqrt_eight_over_pi = np.float32(-1.595769121 * 0.044715)
            squared = values * values
            cubed = squared * values
            exponent = (cubed + values * tanh_approx_factor) * neg_sqrt_eight_over_pi
            exp_neg = _ftz_float32(np.exp(_ftz_float32(exponent)))
            return _div_ftz_float32(values, np.float32(1.0) + exp_neg)
        if activation_type == "gelu_erf":
            if kernel_erf:
                erf = _erf_kernel_float32(values * np.float32(0.707106781))
            else:
                erf = np.vectorize(math.erf, otypes=[np.float32])(
                    values / np.float32(math.sqrt(2))
                )
            return 0.5 * values * (1 + erf)
    raise ValueError("Unsupported activation_type")


def round_bfloat16(values):
    """RNE BF16 boundary used by the activation epilogues, in an FP32 array."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    rounded = values.astype(ml_dtypes.bfloat16).astype(np.float32)
    nan = np.isnan(values)
    if nan.any():
        # Retain the reference's quieted FP32 NaN payload, not a BF16 payload.
        rounded.view(np.uint32)[nan] = values.view(np.uint32)[nan] | np.uint32(
            0x00400000
        )
    return rounded


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
    if (
        activation_type in ("swiglu", "silu")
        and mx_dtype(output_dtype) == "float4_e2m1"
    ):
        raise ValueError("SwiGLU and SiLU require MXFP8")
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
    if isinstance(value, np.ndarray) and "float4_e2m1" in str(value.dtype).lower():
        return _numpy(value)
    if isinstance(value, np.ndarray):
        strides = tuple(stride // value.itemsize for stride in value.strides)
        packed = np.asarray(value, dtype=np.uint8)
    else:
        import torch

        strides = value.stride()
        packed = value.detach().cpu().view(torch.uint8).numpy()
    # Match the Torch bridge's IsTransposeLastTwoDims before any copy can
    # erase the view strides. Unpack in storage order, then restore the view.
    shape = value.shape
    transposed = (
        2 <= len(shape) <= 6
        and strides[-2:] == (1, shape[-2])
        and tuple(shape[-2:]) != (1, 1)
        and strides[:-2] == _contiguous_strides(shape)[:-2]
    )
    if transposed:
        packed = packed.swapaxes(-1, -2)
    nibbles = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    nibbles[..., 0::2] = packed & np.uint8(0x0F)
    nibbles[..., 1::2] = packed >> np.uint8(4)
    decoded = decode_fp4(nibbles)
    return decoded.swapaxes(-1, -2) if transposed else decoded


def _numpy_mx(value, dtype):
    """Decode byte carriers whose MX type is supplied by a Torch dtype attribute."""
    if value is None:
        return None
    if str(value.dtype).removeprefix("torch.") in ("uint8", "int8"):
        raw = value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
        raw = raw.view(np.uint8)
        return decode_e8m0(raw) if dtype == "float8_e8m0" else decode_fp8(raw, dtype)
    return _numpy(value)


def _raw_outputs(raw_y, raw_scale, dtype, torch_output=False, pack_fp4_output=False):
    """Keep native output encodings for the per-output ULP comparison."""
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


def _npu_golden_device(device):
    # Only TTK's explicit NPU request selects the device baseline. Legacy
    # callers may supply device tensors while still requesting a CPU reference.
    target = str(device if device is not None else "cpu")
    return target if target.split(":", 1)[0] == "npu" else None


def _to_npu_tensor(value, dtype, device):
    """Preserve MX encodings, packing NumPy FP4 into Torch's uint8 carrier."""
    import torch

    if value is None:
        return None
    if dtype == "float4_e2m1":
        if isinstance(value, np.ndarray):
            if value.dtype != np.uint8:
                raw = np.ascontiguousarray(value, dtype=np_float4_e2m1).view(np.uint8)
                value = _pack_fp4(raw)
            value = torch.from_numpy(value)
        elif "float4_e2m1fn_x2" in str(value.dtype):
            value = value.view(torch.uint8)
        if value.dtype != torch.uint8:
            raise ValueError("MXFP4 Torch inputs require packed uint8 storage")
        return value.to(device=device)
    torch_dtype = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e8m0": torch.float8_e8m0fnu,
        "float32": torch.float32,
    }[dtype]
    if isinstance(value, np.ndarray):
        if dtype == "float32":
            value = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
        else:
            numpy_dtype = {
                "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
                "float8_e5m2": ml_dtypes.float8_e5m2,
                "float8_e8m0": np_float8_e8m0,
            }[dtype]
            # uint8 carriers already contain MX encodings, not numeric integers.
            array = np.ascontiguousarray(
                value, dtype=np.uint8 if value.dtype == np.uint8 else numpy_dtype
            )
            value = torch.from_numpy(array.view(np.uint8)).view(torch_dtype)
    elif value.dtype == torch.uint8 and dtype != "float32":
        value = value.view(torch_dtype)
    return value.to(device=device, dtype=torch_dtype)


# One prepared weight per worker. Key by content as well as geometry so another
# case/seed cannot reuse stale data. TTK's default warmup prepares NZ outside the
# measured invocation; cold runs intentionally include that preparation.
_NPU_NZ_WEIGHT_CACHE = None


def _npu_nz_weight(value, logical_shape, dtype, device):
    import torch_npu

    global _NPU_NZ_WEIGHT_CACHE
    nd = np.ascontiguousarray(weight_nz_to_nd(value, logical_shape, dtype))
    key = (
        device,
        dtype,
        tuple(logical_shape),
        hashlib.sha256(memoryview(nd).cast("B")).digest(),
    )
    if _NPU_NZ_WEIGHT_CACHE is None or _NPU_NZ_WEIGHT_CACHE[0] != key:
        weight = _to_npu_tensor(nd, dtype, device)
        weight = torch_npu.npu_format_cast(weight, 29)
        if torch_npu.get_npu_format(weight) != 29:
            raise RuntimeError("NPU Golden requires an actual FRACTAL_NZ weight")
        _NPU_NZ_WEIGHT_CACHE = (key, weight)
    return _NPU_NZ_WEIGHT_CACHE[1]


def _npu_activation_golden(
    x1,
    x2,
    bias,
    x1_scale,
    x2_scale,
    *,
    device,
    output_dtype,
    activation_type="silu",
    quant_mode="mx",
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    transpose_x1=False,
    transpose_x2=False,
    x1_dtype=None,
    x2_dtype=None,
    weight_nz_shape=None,
    pack_fp4_output=False,
):
    """MX concatenation baseline shared by kernel, ACLNN and E2E tests."""
    if (
        activation_type not in ("silu", "swiglu", "gelu_tanh", "gelu_erf")
        or quant_mode != "mx"
    ):
        raise NotImplementedError(
            "NPU concatenation Golden supports SiLU/SwiGLU/GELU MX quantization"
        )
    dtype = mx_dtype(output_dtype)
    is_fp4 = dtype == "float4_e2m1"
    if is_fp4:
        if activation_type in ("silu", "swiglu"):
            raise ValueError("SwiGLU and SiLU require MXFP8")
        if round_mode not in ("rint", "floor", "round") or scale_alg not in (0, 2):
            raise ValueError(
                "MXFP4 requires round_mode=rint/floor/round and scale_alg=0/2"
            )
    elif round_mode != "rint" or scale_alg not in (0, 1):
        raise ValueError("MXFP8 requires round_mode=rint and scale_alg=0/1")
    if x1_scale is None or x2_scale is None:
        raise ValueError("MX inputs require explicit x1_scale and x2_scale")
    if (
        weight_nz_shape is not None
        and transpose_x1
        and activation_type in ("silu", "swiglu")
    ):
        raise ValueError("SiLU/SwiGLU WeightNZ requires transpose_x1=false")

    import torch
    import torch.nn.functional as functional
    import torch_npu

    a_dtype, b_dtype = _mx_operand_dtypes(
        x1, x2, dtype, x1_dtype=x1_dtype, x2_dtype=x2_dtype
    )
    a = _to_npu_tensor(x1, a_dtype, device)
    b = (
        _to_npu_tensor(x2, b_dtype, device)
        if weight_nz_shape is None
        else _npu_nz_weight(x2, weight_nz_shape, b_dtype, device)
    )
    a_scale = _to_npu_tensor(x1_scale, "float8_e8m0", device)
    b_scale = _to_npu_tensor(x2_scale, "float8_e8m0", device)
    if transpose_x1:
        a = a.transpose(-1, -2)
        a_scale = a_scale.transpose(-3, -2)
    if transpose_x2:
        b = b.transpose(-1, -2)
        b_scale = b_scale.transpose(-3, -2)
    matrix = torch_npu.npu_quant_matmul(
        a,
        b,
        b_scale,
        pertoken_scale=a_scale,
        bias=_to_npu_tensor(bias, "float32", device),
        output_dtype=torch.bfloat16 if activation_type == "swiglu" else torch.float32,
        group_sizes=[1, 1, 32],
        **({"x1_dtype": 296, "x2_dtype": 296} if is_fp4 else {}),
    )
    if activation_type == "swiglu":
        # SwigluMxQuant takes BF16/FP16 input; use BF16 QBMM output for this
        # performance baseline. Use golden-mode Enable for accuracy checks.
        y, y_scale = torch_npu.npu_swiglu_mx_quant(
            matrix,
            activate_dim=-1,
            activate_left=True,
            swiglu_mode=0,
            axis=-1,
            dst_type={"float8_e4m3fn": 292, "float8_e5m2": 291}[dtype],
            round_mode=round_mode,
            scale_alg=scale_alg,
        )
    else:
        # The fused epilogue rounds AFTER FP32 activation. Keep the Cast in
        # the measured chain instead of rounding the QBMM output prematurely.
        if activation_type == "silu":
            activated = functional.silu(matrix)
        else:
            activated = functional.gelu(
                matrix, approximate="tanh" if activation_type == "gelu_tanh" else "none"
            )
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            activated.to(torch.bfloat16),
            axis=-1,
            block_size=32,
            round_mode=round_mode,
            dst_type=296 if is_fp4 else a.dtype,
            scale_alg=scale_alg,
            # The fused OCP path ignores dst_type_max. DynamicMxQuant validates
            # it unconditionally, so pass its default when that attribute is unused.
            **(
                {"dst_type_max": dst_type_max if scale_alg == 2 else 0.0}
                if is_fp4
                else {}
            ),
        )
    # ACLNN's plugin return converter calls Tensor.numpy() directly, which
    # cannot represent Torch FP8/E8M0. Return native NumPy MX arrays instead.
    raw_y = y.detach().cpu().contiguous().view(torch.uint8).numpy()
    raw_scale = y_scale.detach().cpu().contiguous().view(torch.uint8).numpy()
    if is_fp4:
        raw_y = np.stack((raw_y & 0x0F, raw_y >> 4), axis=-1).reshape(
            raw_y.shape[:-1] + (raw_y.shape[-1] * 2,)
        )
    return _raw_outputs(raw_y, raw_scale, dtype, pack_fp4_output=pack_fp4_output)


_MX_ULP_DTYPES = {
    "float4_e2m1": "float4_e2m1",
    "float4_e2m1fn": "float4_e2m1",
    "uint8": "packed_float4_e2m1",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
    "float8_e8m0": "float8_e8m0",
    "float8_e8m0fnu": "float8_e8m0",
}


@lru_cache(maxsize=4)
def _mx_ulp_table(dtype):
    """ULP distances between codes; -1 denotes a NaN/Inf mismatch."""
    codes = np.arange(256, dtype=np.uint8)
    if dtype == "float8_e8m0":
        ranks = codes.astype(np.int16)
        with np.errstate(over="ignore"):
            values = decode_e8m0(codes)
    else:
        sign = 8 if dtype == "float4_e2m1" else 128
        magnitude = (codes & (sign - 1)).astype(np.int16)
        # Ordered signed ranks merge +/-0 and avoid uint8 subtraction overflow.
        ranks = np.where(codes & sign, -magnitude, magnitude)
        values = decode_fp4(codes) if sign == 8 else decode_fp8(codes, dtype)
    distances = np.abs(ranks[:, None] - ranks[None, :])
    nonfinite = ~np.isfinite(values)
    special = nonfinite[:, None] | nonfinite[None, :]
    same_special = (values[:, None] == values[None, :]) | (
        np.isnan(values[:, None]) & np.isnan(values[None, :])
    )
    distances[special] = -1
    distances[special & same_special] = 0
    return distances


def _compare_mx_output(actual, golden, name):
    """Require at least 99.9% of this output's elements to be within 1 ULP."""
    if actual.shape != golden.shape:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{name}: shape mismatch {actual.shape} != {golden.shape}",
        }
    dtype = _MX_ULP_DTYPES.get(str(golden.dtype).removeprefix("torch."))
    actual_dtype = _MX_ULP_DTYPES.get(str(actual.dtype).removeprefix("torch."))
    if (
        dtype is None
        or actual_dtype != dtype
        or (name == "y_scale") != (dtype == "float8_e8m0")
    ):
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{name}: unsupported or mismatched dtypes {actual.dtype}, {golden.dtype}",
        }
    packed = dtype == "packed_float4_e2m1"
    table = _mx_ulp_table("float4_e2m1" if packed else dtype)
    actual, golden = _mx_output_bytes(actual), _mx_output_bytes(golden)
    elements_per_byte = 2 if packed else 1
    total = actual.size * elements_per_byte
    over_1ulp = nonfinite_mismatches = max_ulp = 0
    diff_indices = []
    for start in range(0, actual.size, _QUANTIZE_CHUNK_ELEMENTS):
        a = actual[start : start + _QUANTIZE_CHUNK_ELEMENTS]
        g = golden[start : start + _QUANTIZE_CHUNK_ELEMENTS]
        if packed:
            # E2E FP4 carries two logical elements per uint8, low nibble first.
            a = np.stack((a & 0x0F, a >> 4), axis=-1).reshape(-1)
            g = np.stack((g & 0x0F, g >> 4), axis=-1).reshape(-1)
        distances = table[a, g]
        over_1ulp += int(np.count_nonzero(distances > 1))
        nonfinite_mismatches += int(np.count_nonzero(distances < 0))
        max_ulp = max(max_ulp, int(np.max(distances, initial=0)))
        if len(diff_indices) < 20:
            indices = np.flatnonzero((distances > 1) | (distances < 0))
            diff_indices.extend(
                (indices[: 20 - len(diff_indices)] + start * elements_per_byte).tolist()
            )
    # Integer arithmetic keeps exactly 1/1000 inclusive, even for small outputs.
    passed = nonfinite_mismatches == 0 and over_1ulp * 1000 <= total
    result = {
        "pass": passed,
        "precision": 100.0 * (1 - (over_1ulp + nonfinite_mismatches) / total)
        if total
        else 100.0,
        "diff_indices": diff_indices,
        "metrics": {
            "max_ulp": max_ulp,
            "over_1ulp_count": over_1ulp,
            "over_1ulp_ratio": over_1ulp / total if total else 0.0,
            "nonfinite_mismatch_count": nonfinite_mismatches,
            "element_count": total,
        },
    }
    if not passed:
        result["error_info"] = (
            f"{name}: {over_1ulp}/{total} elements exceed 1 ULP (limit 0.001); "
            f"NaN/Inf mismatches: {nonfinite_mismatches}"
        )
    return result


def _mx_output_bytes(value):
    """Preserve native codes for NumPy outputs and ACLNN CPU Torch tensors."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).view(np.uint8).reshape(-1)
    import torch

    return value.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)


def compare_mx_outputs(y, y_scale, golden_y, golden_y_scale, **kwargs):
    """Apply the same independent y/y_scale criterion to GELU, SwiGLU and SiLU."""
    return [
        _compare_mx_output(y, golden_y, "y"),
        _compare_mx_output(y_scale, golden_y_scale, "y_scale"),
    ]


class QuantMatmulActivationQuantTestSpec:
    compare = staticmethod(compare_mx_outputs)

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
        device=None,
        **kwargs,
    ):
        dtype = mx_dtype(y_dtype)
        if quant_mode != "mx":
            raise ValueError("This TestSpec covers MX quantization")
        a_dtype, b_dtype = _mx_operand_dtypes(
            x1, x2, dtype, kwargs.get("input_dtypes", ())
        )
        npu_device = _npu_golden_device(device)
        weight_nz_shape = None
        input_formats = kwargs.get("input_formats", ())
        if _is_fractal_nz(input_formats):
            if activation_type in ("swiglu", "silu") and transpose_x1:
                raise ValueError("SwiGLU and SiLU WeightNZ require transpose_x1=false")
            input_ori_shapes = kwargs.get("input_ori_shapes", ())
            if len(input_ori_shapes) <= 1 or input_ori_shapes[1] is None:
                raise ValueError("WeightNZ kernel golden requires input_ori_shapes[1]")
            weight_nz_shape = input_ori_shapes[1]
            if npu_device is None:
                x2 = weight_nz_to_nd(x2, weight_nz_shape, b_dtype)
        if npu_device is not None:
            return _npu_activation_golden(
                x1,
                x2,
                bias,
                x1_scale,
                x2_scale,
                device=npu_device,
                output_dtype=dtype,
                activation_type=activation_type,
                quant_mode=quant_mode,
                round_mode=round_mode,
                scale_alg=scale_alg,
                dst_type_max=dst_type_max,
                transpose_x1=transpose_x1,
                transpose_x2=transpose_x2,
                x1_dtype=a_dtype,
                x2_dtype=b_dtype,
                weight_nz_shape=weight_nz_shape,
            )
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
    compare = staticmethod(compare_mx_outputs)

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
        device=None,
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
        a_dtype, b_dtype = _mx_operand_dtypes(
            x1, x2, dtype, kwargs.get("tensor_dtypes", ())
        )
        npu_device = _npu_golden_device(device)
        weight_nz_shape = None
        if _is_fractal_nz(kwargs.get("tensor_formats", ())):
            if activationType in ("swiglu", "silu") and transposeX1:
                raise ValueError("SwiGLU and SiLU WeightNZ require transpose_x1=false")
            # TTK passes a logical x2 view backed by the physical NZ storage.
            # Recover and decode that storage before _numpy() copies the view.
            weight_nz_shape = tuple(x2.shape)
            if npu_device is None:
                x2 = weight_nz_to_nd(x2, weight_nz_shape, b_dtype)
        if npu_device is not None:
            return _npu_activation_golden(
                x1,
                x2,
                biasOptional,
                x1ScaleOptional,
                x2Scale,
                device=npu_device,
                output_dtype=dtype,
                activation_type=activationType,
                quant_mode=quantMode,
                round_mode=roundMode,
                scale_alg=scaleAlg,
                dst_type_max=dstTypeMax,
                transpose_x1=transposeX1,
                transpose_x2=transposeX2,
                x1_dtype=a_dtype,
                x2_dtype=b_dtype,
                weight_nz_shape=weight_nz_shape,
            )
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
    compare = staticmethod(compare_mx_outputs)

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
        device=None,
        **kwargs,
    ):
        # PyTorch views already express the logical matrix: no second transpose.
        tensor_dtypes = kwargs.get("tensor_dtypes", ())
        dtype = output_dtype
        if dtype is None:
            dtype = x1_dtype
        if dtype is None:
            dtype = tensor_dtypes[0] if tensor_dtypes else x1.dtype
        dtype = mx_dtype(dtype)
        a_dtype, b_dtype = _mx_operand_dtypes(
            x1, x2, dtype, tensor_dtypes, x1_dtype, x2_dtype
        )
        quant_mode = kwargs.get("quant_mode", "mx")
        if quant_mode != "mx":
            raise ValueError("This TestSpec covers MX quantization")
        group_sizes = kwargs.get("group_sizes")
        if group_sizes is not None and tuple(group_sizes) not in (
            (),
            (0, 0, 0),
            (1, 1, 32),
        ):
            raise ValueError("Invalid MX group_sizes")
        scale_alg = 0 if scale_alg is None else scale_alg
        npu_device = _npu_golden_device(device)
        if npu_device is not None:
            if _is_fractal_nz(kwargs.get("tensor_formats", ())):
                import torch_npu

                # Match the NZ binding applied before the actual E2E operator call.
                x2 = torch_npu.npu_format_cast(
                    _to_npu_tensor(x2, b_dtype, npu_device), 29
                )
            return _npu_activation_golden(
                x1,
                x2,
                bias,
                x1_scale,
                x2_scale,
                device=npu_device,
                output_dtype=dtype,
                activation_type=activation_type,
                quant_mode=quant_mode,
                round_mode=round_mode,
                scale_alg=scale_alg,
                dst_type_max=dst_type_max,
                x1_dtype=a_dtype,
                x2_dtype=b_dtype,
                pack_fp4_output=dtype == "float4_e2m1",
            )
        is_fp4 = dtype == "float4_e2m1"
        if is_fp4:
            x1_value = _unpack_fp4(x1)
            x2_value = _unpack_fp4(x2)
        else:
            x1_value = _numpy_mx(x1, a_dtype)
            x2_value = _numpy_mx(x2, b_dtype)
        result = reference(
            x1_value,
            x2_value,
            _numpy(bias),
            _numpy_mx(x1_scale, "float8_e8m0"),
            _numpy_mx(x2_scale, "float8_e8m0"),
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
