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
"""Kernel-level golden for TransposeQuantBatchMatMul (Ascend 950).

op: transpose_quant_batch_mat_mul
formula:
    FP8 (K-C): y = perm_y((perm_x1(x1) @ perm_x2(x2)) * x2_scale[n] * x1_scale[m])
    MXFP8:     y = perm_y(sum_g((perm_x1(x1)[g] * s1[g]) @ (perm_x2(x2)[g] * s2[g])))
    HIFP8:     y = perm_y((perm_x1(x1) @ perm_x2(x2)) * deq(x2_scale)[n])

Precision modes follow the tiling decision tree
(TransposeQuantBatchMatMulAswTiling::DoOpTiling):
    1. IsMicroScaling(x1Scale, x2Scale) -> MXFP8: both scales float8_e8m0,
       4-D layouts [m, b, G, 2] / aligned with perm_x2 on x2, G = ceil(K / 64),
       group_size_k = 32 (hardware MatmulTypeWithScale: scale pre-multiplies
       every 32-element K group before accumulation);
    2. IsHIFP8(x1, x2) -> HIFP8: x1/x2 hifloat8, x2_scale uint64 [n] applied
       per-N column via SetQuantVector in the cube-out fixpipe (x1_scale is
       not read by the kernel);
    3. otherwise -> FP8 (K-C): fp32 vector dequant on AIV, y = (acc * x2_scale)
       * x1_scale (see VFDoDequant).

Ascend 950 constraints (tiling checks, documented for test-case reference):
    - perm_x1: [1, 0, 2] only; perm_y: [1, 0, 2] only;
    - perm_x2: [0, 1, 2] for every mode, [0, 2, 1] only for MXFP8 / HIFP8;
    - FP8 mode: K == 512 and N == 128; MXFP8: K % 64 == 0, x2 may be FRACTAL_NZ
      (only MXFP8), scales e4m3fn x e4m3fn; HIFP8: ND only;
    - batch_split_factor == 1 for all modes; bias is never consumed by the
      kernel (kept in the signature for argument compatibility).
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
from matmul_quant_util import pack_u64_scale, u64_to_deq_scale

# CANN dtype enum -> numpy dtype for the "dtype" attribute (DT_FLOAT16 / DT_BF16).
_DTYPE_ATTR_MAP = {1: np.float16, 27: _util.np_bfloat16}

# Microscaling: fixed 32-element group along K (MX_GROUP_SIZE in tiling).
_MX_GROUP_SIZE = 32


def _is_mxfp8_mode(x1_scale, x2_scale):
    """tiling IsMicroScaling: both scales must exist and be float8_e8m0.

    dtype matching is name-based because test frameworks may supply their own
    float8_e8m0 dtype class (the old golden also matched via "e8m0" substring).
    """
    if x1_scale is None or x2_scale is None:
        return False
    return "e8m0" in _util.dtype_to_str(
        x1_scale.dtype
    ) and "e8m0" in _util.dtype_to_str(x2_scale.dtype)


def _e8m0_to_float(arr):
    """Decode float8_e8m0 to float32: 2^(raw - 127), raw == 255 stays NaN."""
    raw = np.ascontiguousarray(arr).view(np.uint8).astype(np.float32)
    result = np.power(2.0, raw - 127.0)
    result[raw == 255.0] = np.nan
    return result


def _kc_matmul(x1, x2, x1_scale, x2_scale, is_hifp8_mode):
    """matmul + post-scale, matching the AIV VFDoDequant order (acc * s2 * s1)."""
    mm_out = np.matmul(x1, x2)

    if is_hifp8_mode:
        if x2_scale is not None:
            if x2_scale.dtype in (np.int64, np.uint64):
                x2_scale = u64_to_deq_scale(x2_scale)
            mm_out = mm_out * x2_scale.astype(np.float32).reshape(1, 1, -1)
        return mm_out

    if x2_scale is not None:
        mm_out = mm_out * x2_scale.astype(np.float32).reshape(1, 1, -1)
    if x1_scale is not None:
        mm_out = mm_out * x1_scale.astype(np.float32).reshape(1, -1, 1)
    return mm_out


def _mxfp8_matmul(x1, x2, x1_scale, x2_scale, perm_x2):
    """MXFP8: hardware scale pre-multiply per 32-element K group, then matmul.

    x1 / x2 are [b, m, K] / [b, K, n] in float32.
    Scale storage: x1_scale [m, b, G, 2], G = K // 64 (K % 64 == 0 by tiling);
    x2_scale follows the perm_x2 layout of x2: [b, G, n, 2] for (0, 1, 2) and
    [b, n, G, 2] for (0, 2, 1) - the innermost pair covers two adjacent
    32-element K groups within a 64-element block, flattened as sequential
    32-groups along K.
    """
    B, M, K = x1.shape
    N = x2.shape[2]
    num_group = K // 64

    if tuple(perm_x2) == (0, 2, 1):
        x2_scale = np.transpose(x2_scale, (0, 2, 1, 3))  # [b, n, G, 2] -> [b, G, n, 2]

    # [m, b, G, 2] -> [b, m, K]: one scale per 32 K elements, broadcast to K.
    x1_scale_f = _e8m0_to_float(x1_scale).reshape(M, B, num_group * 2)
    s1 = np.transpose(x1_scale_f, (1, 0, 2))
    s1 = np.repeat(s1, _MX_GROUP_SIZE, axis=-1)[..., :K]

    # [b, G, n, 2] -> [b, K, n]: the (g, p) pair covers one 64-element K block,
    # so the pair dim must sit next to G before flattening into sequential
    # 32-element groups (verified against the device on multiple N values).
    x2_scale_f = (
        _e8m0_to_float(x2_scale).transpose(0, 1, 3, 2).reshape(B, num_group * 2, N)
    )
    s2 = np.repeat(x2_scale_f, _MX_GROUP_SIZE, axis=-2)[..., :K, :]

    return np.matmul(x1 * s1, x2 * s2)


def _kernel_compute(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    *,
    dtype=1,
    group_size=0,
    perm_x1=(1, 0, 2),
    perm_x2=(0, 1, 2),
    perm_y=(1, 0, 2),
    batch_split_factor=1,
    **kwargs,
):
    """Core numpy simulation of transpose_quant_batch_mat_mul on Ascend 950.

    bias / group_size / batch_split_factor are accepted but not used, matching
    the kernel: bias is never read, group_size only matters for the MXFP8
    tiling check, and batch_split_factor must be 1.
    """
    is_hifp8_mode = "hifloat8" in _util.dtype_to_str(
        x1.dtype
    ) and "hifloat8" in _util.dtype_to_str(x2.dtype)

    x1_f32 = x1.astype(np.float32)
    x2_f32 = x2.astype(np.float32)

    x1_t = np.transpose(x1_f32, axes=list(perm_x1))
    if tuple(perm_x2) == (0, 2, 1):
        x2_t = np.swapaxes(x2_f32, -2, -1)
    else:
        x2_t = x2_f32

    if _is_mxfp8_mode(x1_scale, x2_scale):
        mm_out = _mxfp8_matmul(x1_t, x2_t, x1_scale, x2_scale, perm_x2)
    else:
        mm_out = _kc_matmul(x1_t, x2_t, x1_scale, x2_scale, is_hifp8_mode)

    mm_out = np.transpose(mm_out, axes=list(perm_y))

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _util.cast_output_dtype(mm_out, output_dtypes[0])
    else:
        mm_out = mm_out.astype(_DTYPE_ATTR_MAP.get(dtype, np.float16))

    return [mm_out]


class TransposeQuantBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        *,
        dtype=1,
        group_size=0,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
        **kwargs,
    ):
        """Kernel golden: FRACTAL_NZ -> ND conversion for x2 (MXFP8 only) + compute."""
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())

        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            if ori_shape is not None and tuple(x2.shape) != tuple(ori_shape):
                x2 = _util.nz_to_nd(x2, ori_shape)

        return _kernel_compute(
            x1,
            x2,
            bias,
            x1_scale,
            x2_scale,
            dtype=dtype,
            group_size=group_size,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            batch_split_factor=batch_split_factor,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        *,
        dtype=1,
        group_size=0,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
        batch_split_factor=1,
        **kwargs,
    ):
        """Input preprocessing aligned with the op input order
        [x1, x2, bias, x1_scale, x2_scale]: uint64/int64 x2_scale (HIFP8
        SetQuantVector layout) is regenerated with a deterministic seed.
        float8_e8m0 scales are passed through unchanged: NaN values (raw 255)
        are intentionally preserved so that the golden reproduces the
        hardware behavior bit-exactly and NaN positions line up with the NPU
        output during comparison. x1_scale is ignored by the HIFP8 kernel and
        therefore never generated.
        """
        if x2_scale is not None and x2_scale.dtype in (np.int64, np.uint64):
            fp32_scale = (
                np.random.RandomState(42)
                .uniform(-5, 5, x2_scale.shape)
                .astype(np.float32)
            )
            x2_scale = pack_u64_scale(fp32_scale).astype(x2_scale.dtype)

        return x1, x2, bias, x1_scale, x2_scale


__spec__ = {
    "transpose_quant_batch_mat_mul": "TransposeQuantBatchMatMulTestSpec",
}
