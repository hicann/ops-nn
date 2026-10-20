# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU references for the Ascend 950 RmsNormDynamicMxQuant interfaces.

Adapted from MyTestCase/Rms_norm_dynamicMXQuant/golden/current and golden/ttk.
The MX exponent, rounding, and minimum-exponent rules are also retained from
golden/historical_0825/golden.py. No TTK or NPU module is imported here.

The native/ACLNN references preserve FP32 normalization intermediates and the
cast back to the original x dtype before MX quantization. Gamma and beta keep
their own input precision. The independent Torch reference uses FP64 RMS/affine
and a separate table-search MX encoder in rms_norm_dynamic_mx_quant_torch.py.

Load this TestSpec with --plugin golden.py --pc=1 --golden-mode Enable.
Promote mode loses the original x dtype and is intentionally rejected.
Kernel and GEIR share ../st/arch35/ttk_kernel_rms_norm_dynamic_mx_quant_st.csv;
select the execution mode explicitly. ACLNN uses its separate interface CSV.
"""

import sys
from pathlib import Path
import numpy as np
from en_dtypes import float4_e1m2, float4_e2m1, float8_e8m0
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2

# TestSpec files are loaded without package context. A named sibling import also
# lets the TTK DATA server discover and sync this dependency through its normal
# missing-module protocol. No Torch or NPU import occurs at module load time.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from rms_norm_dynamic_mx_quant_torch import TorchIndependentReference
finally:
    sys.path.pop(0)

__spec__ = {
    "rms_norm_dynamic_mx_quant": "RmsNormDynamicMxQuantTestSpec",
    "aclnnRmsNormDynamicMxQuant": "AclnnRmsNormDynamicMxQuantTestSpec",
}

_FORMATS = {
    35: (float8_e5m2, 15, 2, -14, 57344.0),
    36: (float8_e4m3fn, 8, 3, -6, 448.0),
    40: (float4_e2m1, 2, 1, 0, 6.0),
    41: (float4_e1m2, 0, 2, 0, 1.75),
}
_BLOCK_SIZE = 32
# Bound temporary FP32/FP64 storage; result arrays still have their full size.
_CHUNK_ELEMENTS = 1 << 20


def _to_numpy(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            import torch

            # Preserve the original BF16 storage without allocating a full
            # FP32 temporary (the unchanged large cases contain billions of
            # elements). NumPy and Torch use the same 16-bit BF16 encoding.
            return value.view(torch.int16).numpy().view(bfloat16)
        return value.numpy()
    return np.asarray(value)


def _round_mantissa(value, mode):
    if mode == "rint":
        return np.rint(value)
    if mode == "floor":
        return np.floor(value)
    if mode == "round":
        return np.copysign(np.floor(np.abs(value) + np.float32(0.5)), value)
    raise ValueError(f"Unsupported round_mode: {mode!r}")


def _power_of_two(exponent):
    # MX exponents are integers. ldexp gives exact binary scaling; FP32 exp2
    # can return 2**127 + 1 ULP and move an FP4 half-way value across its tie.
    finite = np.isfinite(exponent)
    shifts = np.where(finite, exponent, 0).astype(np.int32)
    powers = np.ldexp(np.ones_like(exponent, dtype=np.float32), shifts)
    return np.where(finite, powers, np.float32(np.nan))


def _mx_quantize(value, dst_type, scale_alg, round_mode):
    """Quantize rows; return semantic element and E8M0 arrays, not uint8 values.

    This preserves source subnormals before shared scaling, and follows the
    OCP minimum-shared-exponent flushing, cuBLAS exponent rounding, and the
    unused scale byte's zero encoding. The original archived reference remains
    available for provenance; device validation of these boundaries is required.
    """
    dtype, emax, mantissa_bits, min_exp, max_value = _FORMATS[int(dst_type)]
    shape = value.shape
    length = shape[-1]
    block_count = (length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    scale_shape = shape[:-1] + ((block_count + 1) // 2, 2)
    if value.size == 0:
        return np.empty(shape, dtype=dtype), np.empty(scale_shape, dtype=float8_e8m0)

    source = value.astype(np.float32)
    # Both FP16 and BF16 source subnormals survive widening before scaling.
    # The shared-scale multiply can make them representable FP8 values.
    if length % _BLOCK_SIZE:
        pad = [(0, 0)] * source.ndim
        pad[-1] = (0, block_count * _BLOCK_SIZE - length)
        source = np.pad(source, pad)
    blocks = source.reshape(shape[:-1] + (block_count, _BLOCK_SIZE))
    maximum = np.max(np.abs(blocks), axis=-1, keepdims=True)
    use_ocp = int(scale_alg) == 0 or int(dst_type) in (40, 41)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if use_ocp:
            exponent = np.floor(np.log2(maximum)) - np.float32(emax)
        else:
            # Round the FP32 scale's exponent upwards, matching scale_alg=1.
            raw = np.ascontiguousarray(maximum / np.float32(max_value)).view(np.uint32)
            biased = ((raw >> 23) & 255).astype(np.int16)
            mantissa = raw & np.uint32(0x007FFFFF)
            increment = ((biased > 0) & (biased < 254) & (mantissa > 0)) | (
                (biased == 0) & (mantissa > (1 << 22))
            )
            exponent = (biased + increment.astype(np.int16) - 127).astype(np.float32)
            exponent = np.where(maximum == 0, -np.inf, exponent)

        exponent = np.where(exponent > 127, np.nan, np.maximum(exponent, -127))
        scaled = blocks / _power_of_two(exponent)
        if use_ocp:
            scaled = np.where(
                exponent == -127, np.copysign(np.float32(0), blocks), scaled
            )
        private_exp = np.maximum(
            np.floor(np.log2(np.abs(scaled) + (scaled == 0))), min_exp
        )
        unit = _power_of_two(private_exp - np.float32(mantissa_bits))
        rounded = _round_mantissa(scaled / unit, round_mode) * unit
        rounded = np.clip(rounded, -max_value, max_value)
        # The public boundary contract specifies positive zero for NaN data.
        rounded = np.where(np.isnan(rounded), np.float32(0), rounded)
        elements = rounded.reshape(shape[:-1] + (block_count * _BLOCK_SIZE,))[
            ..., :length
        ]
        elements = elements.astype(dtype)

        codes = np.where(np.isfinite(exponent), exponent + 127, 255)
        scale_bytes = np.zeros(
            shape[:-1] + (2 * ((block_count + 1) // 2),), dtype=np.uint8
        )
        scale_bytes[..., :block_count] = codes[..., 0].astype(np.uint8)
        scales = scale_bytes.reshape(scale_shape).view(float8_e8m0)
    return elements, scales


def _reference(x, gamma, beta, epsilon, scale_alg, round_mode, dst_type, output_rstd):
    x = _to_numpy(x)
    gamma = _to_numpy(gamma)
    beta = _to_numpy(beta)
    if x.dtype.name not in ("float16", "bfloat16"):
        raise ValueError(
            "MX quantization requires the original float16/bfloat16 x dtype; "
            "use --golden-mode Enable. Promoted inputs cannot preserve the T_X cast."
        )
    length = x.shape[-1]
    dtype = _FORMATS[int(dst_type)][0]
    scale_shape = x.shape[:-1] + (((length + 63) // 64), 2)
    y = np.empty(x.shape, dtype=dtype)
    scales = np.empty(scale_shape, dtype=float8_e8m0)
    # Return y, mxscale, rstd in interface order. None excludes the disabled
    # rstd placeholder from comparison; an enabled empty reduction yields NaN.
    rstd = np.empty(x.shape[:-1] + (1,), dtype=np.float32) if output_rstd else None
    if length == 0:
        if rstd is not None:
            rstd.fill(np.nan)
        return y, scales, rstd
    if x.size == 0:
        return y, scales, rstd

    rows = x.size // length
    x_rows = x.reshape(rows, length)
    y_rows = y.reshape(rows, length)
    scale_rows = scales.reshape(rows, -1)
    rstd_rows = rstd.reshape(rows, 1) if rstd is not None else None
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // length)
    # None only permits generation of a reference for the explicit NULL API
    # rejection case. It does not change the product's roundMode validation.
    effective_round_mode = "rint" if round_mode is None else round_mode

    gamma_f = gamma.astype(np.float32)
    beta_f = beta.astype(np.float32) if beta is not None else None

    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        values = x_rows[start:stop].astype(np.float32)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            variance = np.mean(
                values * values, axis=-1, keepdims=True, dtype=np.float32
            )
            inv_np = np.reciprocal(np.sqrt(variance + np.float32(epsilon)))
            normalized_np = (values * inv_np) * gamma_f
            if beta_f is not None:
                normalized_np = normalized_np + beta_f
            normalized_np = normalized_np.astype(x.dtype)
        quantized, mxscale = _mx_quantize(
            normalized_np, dst_type, scale_alg, effective_round_mode
        )
        y_rows[start:stop] = quantized
        scale_rows[start:stop] = mxscale.reshape(stop - start, -1)
        if rstd_rows is not None:
            rstd_rows[start:stop] = inv_np
    return y, scales, rstd


def rms_norm_dynamic_mx_quant_golden(
    x,
    gamma,
    beta=None,
    epsilon=1e-6,
    scale_alg=0,
    round_mode="rint",
    dst_type=40,
    output_rstd=False,
    **kwargs,
):
    return _reference(
        x, gamma, beta, epsilon, scale_alg, round_mode, dst_type, output_rstd
    )


def aclnn_rms_norm_dynamic_mx_quant_golden(
    x,
    gamma,
    beta,
    epsilon,
    scale_alg,
    round_mode,
    dst_type,
    output_rstd,
    y_out=None,
    mxscale_out=None,
    rstd_out=None,
    **kwargs,
):
    # Positional order follows GetWorkspaceSize, with pure output tensors skipped
    # by TTK. Names use TTK's snake_case attribute convention.
    return _reference(
        x, gamma, beta, epsilon, scale_alg, round_mode, dst_type, output_rstd
    )


class RmsNormDynamicMxQuantTestSpec:
    golden = rms_norm_dynamic_mx_quant_golden
    third_party = {"torch": TorchIndependentReference}
    # FP4/E8M0 require exact bytes. stat_rel_err uses the community thresholds:
    # E4M3FN 2^-3, E5M2 2^-2, FP32 2^-13; MERE < threshold and MARE < 10x it.
    # TestSpec tolerance takes precedence over CSV; do not override it via CLI.
    tolerance = {
        "float4_e2m1": {"standard": "binary_equal"},
        "float4_e1m2": {"standard": "binary_equal"},
        "float8_e8m0": {"standard": "binary_equal"},
        "float8_e4m3fn": {"standard": "stat_rel_err"},
        "float8_e5m2": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


class AclnnRmsNormDynamicMxQuantTestSpec(RmsNormDynamicMxQuantTestSpec):
    golden = aclnn_rms_norm_dynamic_mx_quant_golden
