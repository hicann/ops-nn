# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU Torch MX encoder using format tables and interval searches.

No NumPy quantizer is called. NumPy dtype views only materialize the public
semantic storage for TTK serialization. RMS/affine are handled separately.
"""

import numpy as np
from en_dtypes import float4_e1m2, float4_e2m1, float8_e8m0
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2

_FORMATS = {
    35: (float8_e5m2, 2, 15, 15, 57344.0, 123, 128),
    36: (float8_e4m3fn, 3, 7, 8, 448.0, 126, 128),
    40: (float4_e2m1, 1, 1, 2, 6.0, 7, 8),
    41: (float4_e1m2, 2, 1, 0, 1.75, 7, 8),
}
_UNSET = object()


def torch_mx_quantize(value, dst_type, scale_alg, round_mode):
    import torch

    dtype, mantissa_bits, bias, emax, max_value, max_code, sign_code = _FORMATS[
        int(dst_type)
    ]
    shape = value.shape
    length = shape[-1]
    blocks = (length + 31) // 32
    scale_shape = shape[:-1] + ((blocks + 1) // 2, 2)
    if value.size == 0:
        return np.empty(shape, dtype=dtype), np.empty(scale_shape, dtype=float8_e8m0)
    source = torch.from_numpy(value.astype(np.float64))
    # Preserve source subnormals through widening and shared scaling.
    if length % 32:
        source = torch.nn.functional.pad(source, (0, blocks * 32 - length))
    source = source.reshape(shape[:-1] + (blocks, 32))
    maximum = source.abs().amax(dim=-1, keepdim=True)
    ocp = int(scale_alg) == 0 or int(dst_type) in (40, 41)
    if ocp:
        mantissa, exponent = torch.frexp(maximum)
        exponent = exponent.to(torch.float64) - 1 - emax
        exponent = torch.where(maximum == 0, -torch.inf, exponent)
    else:
        # The scale is defined in FP32; frexp determines its ceil-log2
        # independently of the NumPy encoder's bit-field implementation.
        factor = maximum.to(torch.float32) / max_value
        mantissa, exponent = torch.frexp(factor)
        exponent = exponent.to(torch.float64) - (mantissa == 0.5).to(torch.float64)
        exponent = torch.where(factor == 0, -torch.inf, exponent)
    exponent = torch.maximum(exponent, torch.tensor(-127.0, dtype=torch.float64))
    exponent = torch.where(
        (maximum.isinf() | maximum.isnan() | (exponent > 127)), torch.nan, exponent
    )
    scaled = source / torch.exp2(exponent)
    if ocp:
        scaled = torch.where(
            exponent == -127, torch.copysign(torch.zeros_like(source), source), scaled
        )
    codes = torch.arange(max_code + 1, dtype=torch.int64)
    raw_exp = codes >> mantissa_bits
    raw_mantissa = codes & ((1 << mantissa_bits) - 1)
    levels = torch.where(
        raw_exp == 0,
        raw_mantissa.to(torch.float64) * (2.0 ** (1 - bias - mantissa_bits)),
        (1 + raw_mantissa.to(torch.float64) / (1 << mantissa_bits))
        * torch.exp2(raw_exp.to(torch.float64) - bias),
    )
    magnitude = scaled.abs().contiguous()
    if round_mode in ("rint", "round"):
        midpoints = (levels[:-1] + levels[1:]) * 0.5
        indices = torch.bucketize(magnitude, midpoints, right=round_mode == "round")
        if round_mode == "rint":
            safe = indices.clamp(max=midpoints.numel() - 1)
            ties = (indices < midpoints.numel()) & (magnitude == midpoints[safe])
            indices = indices + (ties & ((indices & 1) != 0)).to(torch.int64)
    elif round_mode == "floor":
        lower = (torch.bucketize(magnitude, levels, right=True) - 1).clamp(0, max_code)
        upper = torch.bucketize(magnitude, levels, right=False).clamp(0, max_code)
        indices = torch.where(torch.signbit(scaled), upper, lower)
    else:
        raise ValueError(f"Unsupported round mode: {round_mode}")
    raw = indices.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) * sign_code)
    raw = torch.where(scaled.isnan(), torch.zeros_like(raw), raw)
    y = (
        raw.reshape(shape[:-1] + (blocks * 32,))[..., :length]
        .contiguous()
        .numpy()
        .view(dtype)
    )
    scale_codes = (
        torch.where(exponent.isfinite(), exponent + 127, 255)
        .to(torch.uint8)
        .squeeze(-1)
    )
    scales = torch.zeros(shape[:-1] + (2 * ((blocks + 1) // 2),), dtype=torch.uint8)
    scales[..., :blocks] = scale_codes
    return y, scales.reshape(scale_shape).numpy().view(float8_e8m0)


class TorchIndependentReference:
    """FP64 Torch RMS/affine plus independently table-decoded MX encoding."""

    def __call__(
        self,
        x,
        gamma,
        *,
        beta=None,
        epsilon=1e-6,
        scale_alg=0,
        round_mode="rint",
        dst_type=40,
        output_rstd=False,
        scaleAlg=_UNSET,
        roundMode=_UNSET,
        dstType=_UNSET,
        outputRstd=_UNSET,
        **kwargs,
    ):
        import torch

        if scaleAlg is not _UNSET:
            scale_alg = scaleAlg
        if roundMode is not _UNSET:
            round_mode = roundMode
        if dstType is not _UNSET:
            dst_type = dstType
        if outputRstd is not _UNSET:
            output_rstd = outputRstd
        name = str(x.dtype).replace("torch.", "")
        if name not in ("float16", "bfloat16"):
            raise ValueError("Independent MX reference requires the original x dtype")
        storage_dtype = np.float16 if name == "float16" else bfloat16

        def tensor64(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().to(torch.float64)
            return torch.from_numpy(np.asarray(value).astype(np.float64))

        shape = tuple(x.shape)
        n = shape[-1]
        y = np.empty(shape, _FORMATS[int(dst_type)][0])
        scales = np.empty(shape[:-1] + ((n + 63) // 64, 2), float8_e8m0)
        rstd = np.empty(shape[:-1] + (1,), np.float32) if output_rstd else None
        if n == 0:
            if rstd is not None:
                rstd.fill(np.nan)
            return y, scales, rstd
        if y.size == 0:
            return y, scales, rstd
        x_rows = (
            x.detach().cpu().reshape(-1, n)
            if isinstance(x, torch.Tensor)
            else np.asarray(x).reshape(-1, n)
        )
        gt = tensor64(gamma)
        bt = None if beta is None else tensor64(beta)
        y_rows, s_rows = y.reshape(-1, n), scales.reshape(-1, 2 * ((n + 63) // 64))
        r_rows = None if rstd is None else rstd.reshape(-1, 1)
        step = max(1, (1 << 20) // n)
        for begin in range(0, x_rows.shape[0], step):
            stop = min(begin + step, x_rows.shape[0])
            values = tensor64(x_rows[begin:stop])
            inv = torch.rsqrt(
                torch.mean(values * values, dim=-1, keepdim=True) + float(epsilon)
            )
            norm = (values * inv) * gt
            if bt is not None:
                norm = norm + bt
            # This is a storage cast, not a shared normalization or encoder.
            normalized = norm.numpy().astype(storage_dtype)
            yy, ss = torch_mx_quantize(
                normalized, dst_type, scale_alg, round_mode or "rint"
            )
            y_rows[begin:stop] = yy
            s_rows[begin:stop] = ss.reshape(stop - begin, -1)
            if r_rows is not None:
                r_rows[begin:stop] = inv.numpy().astype(np.float32)
        return y, scales, rstd
