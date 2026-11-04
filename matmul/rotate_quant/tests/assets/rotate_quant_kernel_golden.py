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
"""
rotate_quant kernel golden 实现。

计算流程:
1. 旋转变换: Y = (x.reshape(-1, K) @ rotation).reshape(x.shape), rotation 支持 (K,K) 或 (N/K,K,K)
2. 矩阵乘中间精度 bfloat16 (Fixpipe F322BF16)
3. 可选 clamp: alpha 在 (0,1) 内时, 按 32 一组求 groupMax, Y clamp 到 ±alpha*groupMax
4. MX 量化: group=32, scale=e8m0, 输出 y + scale
"""

import ast
from functools import lru_cache
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util

sys.path.insert(0, os.path.dirname(__file__))

np_bfloat16 = _util.np_bfloat16
np_fp4_e2m1 = _util.np_fp4_e2m1
np_fp8_e4m3 = _util.np_fp8_e4m3
np_fp8_e5m2 = _util.np_fp8_e5m2
np_mx_scale = _util.np_fp8_e8m0

MXFP4_DST_DTYPE = 40
FP8_E4M3FN_DST_DTYPE = 36
FP8_E5M2_DST_DTYPE = 35

_MX_DTYPE_MAP = {
    "float4_e2m1": np_fp4_e2m1,
    "float8_e4m3fn": np_fp8_e4m3,
    "float8_e5m2": np_fp8_e5m2,
}

_Y_DTYPE_TO_MX = {
    MXFP4_DST_DTYPE: "float4_e2m1",
    FP8_E4M3FN_DST_DTYPE: "float8_e4m3fn",
    FP8_E5M2_DST_DTYPE: "float8_e5m2",
}


# Shared output checks for the Kernel, ACLNN and Torch adapters.
def _as_numpy(value):
    if hasattr(value, "detach"):
        return _util.torch_to_numpy(value)
    return np.asarray(value)


def _first_differences(mismatch):
    flat = mismatch.reshape(-1)
    indices = []
    for start in range(0, flat.size, 65536):
        positions = np.flatnonzero(flat[start : start + 65536])[: 8 - len(indices)]
        indices.extend((positions + start).tolist())
        if len(indices) == 8:
            break
    return [
        list(map(int, np.unravel_index(index, mismatch.shape))) for index in indices
    ]


def _result(mismatch, *, allowed_fraction=0.0, reason="", **metrics):
    count = int(np.count_nonzero(mismatch))
    size = mismatch.size
    result = {
        "pass": count <= allowed_fraction * size,
        "precision": 100.0 * (size - count) / size if size else 100.0,
        "metrics": {
            "checked_elements": int(size),
            "bad_elements": count,
            "allowed_bad_fraction": allowed_fraction,
            **metrics,
        },
    }
    if not result["pass"]:
        result["error_info"] = reason or "RotateQuant output exceeds its tolerance"
        result["diff_indices"] = _first_differences(mismatch)
    return result


def _shape_dtype_failure(output, golden):
    if output.shape == golden.shape and output.dtype == golden.dtype:
        return None
    return {
        "pass": False,
        "precision": 0.0,
        "error_info": (
            f"shape/dtype mismatch: actual={output.shape}/{output.dtype}, "
            f"golden={golden.shape}/{golden.dtype}"
        ),
    }


def _allowed_bad_fraction(compare_context):
    fields = (
        getattr(compare_context, "csv_fields", {})
        if compare_context is not None
        else {}
    )
    tolerances = fields.get("precision_tolerances", "")
    if isinstance(tolerances, str):
        tolerances = ast.literal_eval(tolerances) if tolerances else ()
    fraction = float(tolerances[0][1]) if tolerances else 0.0
    if not 0 <= fraction <= 1:
        raise ValueError(
            "precision_tolerances must specify a failure fraction in [0, 1]"
        )
    return fraction


@lru_cache(maxsize=4)
def _finite_values(dtype_name):
    """Sorted representable values; adjacent entries are one target-format ULP apart."""
    if dtype_name in ("float4_e2m1", "float4_e2m1fn"):
        positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    elif dtype_name in ("float8_e4m3fn", "float8_e5m2"):
        mantissa_bits, bias, max_code = (
            (3, 7, 126) if dtype_name == "float8_e4m3fn" else (2, 15, 123)
        )
        code = np.arange(max_code + 1, dtype=np.int32)
        exponent = code >> mantissa_bits
        fraction = (code & ((1 << mantissa_bits) - 1)) / float(1 << mantissa_bits)
        positive = np.where(
            exponent == 0,
            fraction * 2.0 ** (1 - bias),
            (1 + fraction) * np.exp2(exponent - bias),
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported RotateQuant output dtype: {dtype_name}")
    return np.concatenate((-positive[:0:-1], positive))


def compare_quantized(output, golden, compare_context=None):
    output, golden = _as_numpy(output), _as_numpy(golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    values = _finite_values(golden.dtype.name)
    actual, expected = output.astype(np.float32), golden.astype(np.float32)
    # Single-sided NaN and differing infinities always fail, regardless of tolerance.
    special_mismatch = (np.isnan(actual) != np.isnan(expected)) | (
        (np.isinf(actual) | np.isinf(expected)) & (actual != expected)
    )
    if np.any(special_mismatch):
        return _result(special_mismatch, reason="Y NaN/Inf positions or signs differ")
    finite = np.isfinite(actual) & np.isfinite(expected)
    actual_rank = np.searchsorted(values, actual[finite])
    expected_rank = np.searchsorted(values, expected[finite])
    mismatch = np.zeros(actual.shape, dtype=bool)
    distances = np.abs(actual_rank - expected_rank)
    mismatch[finite] = distances > 1
    return _result(
        mismatch,
        allowed_fraction=_allowed_bad_fraction(compare_context),
        reason="Y exceeds one adjacent representable value",
        max_adjacent_distance=int(distances.max()) if distances.size else 0,
    )


def compare_codes(output, golden):
    output, golden = _as_numpy(output), _as_numpy(golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    if output.dtype.itemsize != 1:
        raise ValueError("RotateQuant encoded outputs must use one-byte NumPy carriers")
    return _result(
        np.ascontiguousarray(output).view(np.uint8)
        != np.ascontiguousarray(golden).view(np.uint8),
        reason="RotateQuant output codes differ",
    )


def compare_scale(output, golden):
    output, golden = _as_numpy(output), _as_numpy(golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    if "float8_e8m0" not in golden.dtype.name or golden.dtype.itemsize != 1:
        raise ValueError("RotateQuant scales must retain their E8M0 dtype")
    # Includes padding and NaN codes. A code delta of one is never tolerated.
    return compare_codes(output, golden)


def grouped_ranges(group_list, group_list_type, row_count, group_count=None):
    """Validate count/cumsum metadata and return each group's [start,end) rows."""
    groups = _as_numpy(group_list)
    if groups.ndim != 1 or groups.dtype != np.dtype("int64"):
        raise ValueError("group_list must be a one-dimensional int64 tensor")
    if not 1 <= groups.size <= 1024 or (
        group_count is not None and groups.size != group_count
    ):
        raise ValueError("group count must be in [1,1024] and match rotation dim0")
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1")
    ranges = []
    start = 0
    for value in groups:
        value = int(value)
        end = value if group_list_type == 0 else start + value
        if value < 0 or end < start or end > row_count:
            raise ValueError("Invalid group boundary")
        ranges.append((start, end))
        start = end
    return ranges


def _covered_rows(compare_context, row_count):
    if compare_context is None:
        return None
    inputs = getattr(compare_context, "input_tensors", ())
    # Kernel/Torch have four tensor slots; ACLNN V2 adds two output slots.
    if inputs is None or len(inputs) not in (4, 6) or inputs[3] is None:
        return None
    attrs = getattr(compare_context, "attributes", {})
    mode = attrs.get("group_list_type", attrs.get("groupListType", 0))
    return grouped_ranges(inputs[3], mode, row_count)[-1][1]


def compare_grouped_outputs(
    y, scale, golden_y, golden_scale, compare_context, *, preserve_tail
):
    """Check computed rows with MX tolerances and caller-owned tails exactly.

    Torch allocates empty outputs, so its uncovered tail has no defined initial
    value. Kernel/ACLNN buffers are initialized by the caller and must retain it.
    """
    arrays = [_as_numpy(value) for value in (y, scale, golden_y, golden_scale)]
    y, scale, golden_y, golden_scale = arrays
    failures = [
        _shape_dtype_failure(y, golden_y),
        _shape_dtype_failure(scale, golden_scale),
    ]
    if any(failure is not None for failure in failures):
        return [
            failure if failure is not None else {"pass": True, "precision": 100.0}
            for failure in failures
        ]
    covered = _covered_rows(compare_context, golden_y.shape[0])
    if covered is None:
        return [
            compare_quantized(y, golden_y, compare_context),
            compare_scale(scale, golden_scale),
        ]
    y_result = compare_quantized(y[:covered], golden_y[:covered], compare_context)
    scale_result = compare_scale(scale[:covered], golden_scale[:covered])
    for actual, expected, result in (
        (y, golden_y, y_result),
        (scale, golden_scale, scale_result),
    ):
        if preserve_tail:
            tail = compare_codes(actual[covered:], expected[covered:])
            if not tail["pass"]:
                result["pass"] = False
                result["precision"] = min(result["precision"], tail["precision"])
                result["error_info"] = "Uncovered output rows were modified"
                result["diff_indices"] = [
                    [index[0] + covered, *index[1:]] for index in tail["diff_indices"]
                ]
            result.setdefault("metrics", {})["tail_bad_elements"] = tail["metrics"][
                "bad_elements"
            ]
        result.setdefault("metrics", {}).update(
            {
                "covered_rows": covered,
                "uncovered_rows": golden_y.shape[0] - covered,
                "tail_policy": "exact" if preserve_tail else "unspecified",
            }
        )
    return [y_result, scale_result]


def compare_outputs(*outputs, compare_context=None, **kwargs):
    """TTK order: y, scale, golden_y, golden_scale."""
    if len(outputs) != 4:
        raise ValueError(
            "RotateQuant comparison requires two actual and two golden outputs"
        )
    return compare_grouped_outputs(*outputs, compare_context, preserve_tail=True)


# ============================================================================
# MX quantize (BF16 reference for the RotateQuant NPU epilogue)
# ============================================================================


def _mx_scale_and_reciprocal(group_max, mx_ele_dtype, scale_alg, dst_type_max):
    """Mirror ScaleOcpVf / ScaleDynVf / ScaleCublasVf in arch35/mx_quant.h.

    The reciprocal is a BF16 *bit pattern*, not generally 1 / decoded E8M0.
    In particular, OCP/Dyn clear it for code 0 and special-case code 254;
    cuBLAS retains the unsigned-subtraction result at both endpoints.
    """
    max_bits = group_max.astype(np_bfloat16).view(np.uint16)
    max_exp = max_bits & np.uint16(0x7F80)
    finite = max_exp != 0x7F80
    if scale_alg == 0 or (scale_alg == 2 and dst_type_max in (6, 7)):
        if scale_alg == 0:
            emax = {"float4_e2m1": 2, "float8_e4m3fn": 8, "float8_e5m2": 15}[
                mx_ele_dtype
            ]
            shared_bits = np.maximum(max_exp, emax << 7) - (emax << 7)
        else:
            add_bits = 0x1F if dst_type_max == 7 else 0x3F
            rounded = (max_bits.astype(np.uint32) + add_bits) & 0x7F80
            rounded = np.where(max_exp < 0x0100, 0x0100, rounded)
            shared_bits = rounded - 0x0100
        codes = np.where(finite, shared_bits >> 7, 255).astype(np.uint8)
        reciprocal = (0x7F00 - shared_bits).astype(np.uint16)
        reciprocal = np.where(finite, reciprocal, 0x7F81)
        reciprocal = np.where(shared_bits != 0, reciprocal, 0)
        reciprocal = np.where(shared_bits == 0x7F00, 0x0040, reciprocal)
    else:
        # Host computes invDstTypeMax in FP32. Replacing this multiply with
        # division can change the exponent-round-up decision at a boundary.
        max_norm = {"float8_e4m3fn": 448, "float8_e5m2": 57344}.get(
            mx_ele_dtype, dst_type_max
        )
        inv_max = np.float32(1) / np.float32(max_norm)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_max = group_max.astype(np.float32) * inv_max
        bits = scaled_max.view(np.uint32)
        exponent = bits >> 23
        mantissa = bits & 0x007FFFFF
        round_up = (bits > 0x00400000) & (bits < 0x7F000000) & (mantissa != 0)
        if mx_ele_dtype == "float4_e2m1":
            round_up &= (
                exponent > 0
            )  # RotateQuant FP4 does not round subnormal scales up.
        code = exponent + round_up.astype(np.uint32)
        code = np.where(finite, code, 255)
        code = np.where(max_bits != 0, code, 0)
        codes = code.astype(np.uint8)
        reciprocal = (0x7F00 - (code << 7)).astype(np.uint16)
        reciprocal = np.where(finite, reciprocal, 0x7F81)
        reciprocal = np.where(max_bits != 0, reciprocal, 0)
    return codes, reciprocal.astype(np.uint16).view(np_bfloat16).astype(np.float32)


def _quantize_fp4(values, round_mode):
    """BF16 -> FP4 with the kernel's RINT / ROUND / FLOOR conversion."""
    values = np.clip(values, -6, 6)
    finite_abs = np.where(np.isfinite(values) & (values != 0), np.abs(values), 1)
    exponent = np.maximum(np.floor(np.log2(finite_abs)), 0)
    step = np.exp2(exponent - 1)
    units = values / step
    if round_mode == "rint":
        rounded = np.rint(units)
    elif round_mode == "round":
        rounded = np.copysign(np.floor(np.abs(units) + 0.5), units)
    else:
        rounded = np.floor(units)
    # FP4 has no NaN encoding; keep the existing zero convention for NaNs.
    return np.nan_to_num(np.clip(rounded * step, -6, 6), nan=0).astype(np_fp4_e2m1)


def mx_quantize(
    fp_array: np.ndarray,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0,
):
    """Reference the BF16 RotateQuant epilogue, including E8M0 code padding.

    Inputs are narrowed to the Fixpipe BF16 format before the vector stages.
    All paths multiply by the BF16 reciprocal and round the product to BF16
    before the final FP4 or saturating FP8 conversion.
    """
    if not isinstance(fp_array, np.ndarray) or fp_array.ndim == 0:
        raise ValueError("MX input must be an ndarray with at least one dimension")
    if mx_ele_dtype not in _MX_DTYPE_MAP:
        raise ValueError(f"Unsupported MX dtype: {mx_ele_dtype}")
    if axis not in (-1, fp_array.ndim - 1) or block_size != 32:
        raise ValueError("RotateQuant requires the last axis and block_size=32")
    if round_mode not in ("rint", "round", "floor"):
        raise ValueError(f"Unsupported RotateQuant round_mode: {round_mode}")
    allowed = (0, 2) if mx_ele_dtype == "float4_e2m1" else (0, 1)
    if scale_alg not in allowed:
        raise ValueError(f"Unsupported scale_alg={scale_alg} for {mx_ele_dtype}")
    if mx_ele_dtype != "float4_e2m1" and round_mode != "rint":
        raise ValueError("RotateQuant FP8 only supports rint")
    dst_type_max = float(np.float32(dst_type_max))
    if not (6 <= dst_type_max <= 12 if scale_alg == 2 else dst_type_max == 0):
        raise ValueError("dst_type_max must be in [6,12] for scale_alg=2, otherwise 0")

    shape = fp_array.shape
    n = shape[-1]
    groups = (n + 31) // 32
    values = fp_array.astype(np_bfloat16).astype(np.float32)
    if n % 32:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, groups * 32 - n)])
    blocks = values.reshape(shape[:-1] + (groups, 32))
    group_max = np.max(np.abs(blocks), axis=-1, keepdims=True)
    codes, reciprocal = _mx_scale_and_reciprocal(
        group_max, mx_ele_dtype, scale_alg, dst_type_max
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = (blocks * reciprocal).astype(np_bfloat16).astype(np.float32)
    if mx_ele_dtype == "float4_e2m1":
        quantized = _quantize_fp4(scaled, round_mode)
    else:
        max_norm = 448 if mx_ele_dtype == "float8_e4m3fn" else 57344
        # The kernel uses SatMode::SAT: infinities clamp and NaNs become zero.
        saturated = np.nan_to_num(scaled, nan=0, posinf=max_norm, neginf=-max_norm)
        quantized = np.clip(saturated, -max_norm, max_norm).astype(
            _MX_DTYPE_MAP[mx_ele_dtype]
        )
    y = quantized.reshape(shape[:-1] + (groups * 32,))[..., :n]
    codes = codes.squeeze(-1)
    if groups % 2:
        codes = np.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, 1)], constant_values=0)
    scale = codes.reshape(shape[:-1] + ((groups + 1) // 2, 2)).view(np_mx_scale)
    return y, scale


# ============================================================================
# rotate_quant compute functions
# ============================================================================


def _to_numpy(t):
    """torch.Tensor / ndarray → numpy (bf16 等 torch-only dtype 转 ml_dtypes)."""
    if t is None or isinstance(t, np.ndarray):
        return t
    return _util.torch_to_numpy(t) if hasattr(t, "detach") else np.asarray(t)


def _apply_clamp(y_bf16, alpha):
    """alpha 在 (0.0, 1.0) 开区间内时, 按 32 一组 clamp 到 ±alpha*groupMax (bf16 域)."""
    a = float(np.asarray(alpha).reshape(-1)[0])
    if not (0.0 < a < 1.0):
        return y_bf16
    orig_shape = y_bf16.shape
    n = orig_shape[-1]
    rows = int(np.prod(orig_shape[:-1]))
    y = y_bf16.astype(np.float32).reshape(rows, n)
    g = -(-n // 32)
    y_pad = y.reshape(rows * g, 32)
    group_max = np.max(np.abs(y_pad), axis=-1, keepdims=True)
    limit = (group_max * np.float32(a)).astype(np_bfloat16).astype(np.float32)
    y_pad = np.clip(y_pad, -limit, limit)
    return y_pad.reshape(orig_shape).astype(np_bfloat16)


def compute_rotate_mx(
    x,
    rotation,
    alpha=None,
    mx_ele_dtype="float4_e2m1",
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0,
):
    """旋转 + 可选 clamp + MX 量化."""
    x_f = _to_numpy(x).astype(np.float32)
    rot_f = _to_numpy(rotation).astype(np.float32)
    orig_shape = x_f.shape
    k = rot_f.shape[-1]
    if rot_f.ndim == 3:
        blocks = rot_f.shape[0]
        rows = int(np.prod(orig_shape[:-1]))
        xb = x_f.reshape(rows, blocks, 1, k)
        y = np.matmul(xb, rot_f[None, :, :, :]).reshape(orig_shape)
    else:
        y = np.matmul(x_f.reshape(-1, k), rot_f).reshape(orig_shape)
    y = y.astype(np_bfloat16)
    if alpha is not None:
        y = _apply_clamp(y, _to_numpy(alpha))
    ele_array, scale_array = mx_quantize(
        y,
        mx_ele_dtype=mx_ele_dtype,
        axis=-1,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
    return [ele_array, scale_array]


def compute_grouped_rotate_mx(
    x,
    rotation,
    alpha=None,
    group_list=None,
    group_list_type=0,
    mx_ele_dtype="float4_e2m1",
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0,
    initial_y=None,
    initial_scale=None,
):
    """Shared grouped reference; explicit initial outputs preserve partial tails."""
    x, rotation = _to_numpy(x), _to_numpy(rotation)
    if x.ndim != 2 or rotation.ndim not in (3, 4):
        raise ValueError(
            "Grouped inputs require x[M,N] and rotation[E,K,K] or [E,N/K,K,K]"
        )
    m, n = x.shape
    k = rotation.shape[-1]
    if k not in (32, 64, 128, 256) or rotation.shape[-2] != k or n <= 0 or n % k:
        raise ValueError("Invalid grouped rotation dimensions")
    if rotation.ndim == 4 and rotation.shape[1] != n // k:
        raise ValueError("Grouped 4D rotation dim1 must equal N/K")
    ranges = grouped_ranges(group_list, group_list_type, m, rotation.shape[0])
    # Validate quantization attributes even when every group is empty.
    compute_rotate_mx(
        x[:0],
        rotation[0],
        alpha,
        mx_ele_dtype,
        round_mode,
        scale_alg,
        dst_type_max,
    )
    y_dtype = _MX_DTYPE_MAP[mx_ele_dtype]
    scale_shape = (m, (n + 63) // 64, 2)
    y = (
        np.ones(x.shape, dtype=y_dtype)
        if initial_y is None
        else _as_numpy(initial_y).copy()
    )
    scale = (
        np.ones(scale_shape, dtype=np_mx_scale)
        if initial_scale is None
        else _as_numpy(initial_scale).copy()
    )
    if y.shape != x.shape or y.dtype != np.dtype(y_dtype):
        raise ValueError("Initial grouped Y has the wrong shape or dtype")
    if scale.shape != scale_shape or scale.dtype != np.dtype(np_mx_scale):
        raise ValueError("Initial grouped scale has the wrong shape or dtype")
    for index, (start, end) in enumerate(ranges):
        if end > start:
            y[start:end], scale[start:end] = compute_rotate_mx(
                x[start:end],
                rotation[index],
                alpha,
                mx_ele_dtype,
                round_mode,
                scale_alg,
                dst_type_max,
            )
    return [y, scale]


# ============================================================================
# TestSpec
# ============================================================================


class RotateQuantTestSpec:
    """rotate_quant 算子的 TestSpec 类。"""

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        """多输出精度比对：ttk 传入顺序为 npu0, npu1, ..., golden0, golden1, ..."""
        return compare_outputs(*outputs, compare_context=compare_context, **kwargs)

    @staticmethod
    def golden(
        x,
        rotation,
        alpha=None,
        group_list=None,
        *,
        group_list_type=0,
        y_dtype=MXFP4_DST_DTYPE,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        trans=False,
        **kwargs,
    ):
        if int(y_dtype) not in _Y_DTYPE_TO_MX:
            raise ValueError(f"Unsupported RotateQuant y_dtype: {y_dtype}")
        if axis not in (-1, x.ndim - 1) or trans:
            raise ValueError("RotateQuant requires the last axis and trans=False")
        mx_ele_dtype = _Y_DTYPE_TO_MX[int(y_dtype)]
        if group_list is not None:
            return compute_grouped_rotate_mx(
                x,
                rotation,
                alpha,
                group_list,
                group_list_type,
                mx_ele_dtype,
                str(round_mode),
                int(scale_alg),
                float(dst_type_max),
            )
        return compute_rotate_mx(
            x,
            rotation,
            alpha=alpha,
            mx_ele_dtype=mx_ele_dtype,
            round_mode=str(round_mode),
            scale_alg=int(scale_alg),
            dst_type_max=float(dst_type_max),
        )


__spec__ = {
    "rotate_quant": "RotateQuantTestSpec",
}
