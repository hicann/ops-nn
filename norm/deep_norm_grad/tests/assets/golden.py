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
"""Torch CPU input and golden plugin for DeepNormGrad."""

import numpy as np
import torch
import ml_dtypes  # noqa: F401 - registers bfloat16 with NumPy


__golden__ = {"kernel": {"deep_norm_grad": "deep_norm_grad_golden"}}
__input__ = {"kernel": {"deep_norm_grad": "deep_norm_grad_input"}}

VL_FP32 = 64
DICHOTOMY_ADD_COEFF = 2
MAX_TILE_LENGTH = 4096


def _flatten_shape(x, gamma):
    cols = int(np.prod(_shape_of(gamma), dtype=np.int64))
    if cols == 0:
        return 0, 0
    rows = _numel(x) // cols
    return rows, cols


def _shape_of(value):
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _numel(value):
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    return int(np.asarray(value).size)


def _numpy_dtype(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            return np.dtype(ml_dtypes.bfloat16)
        if value.dtype == torch.float16:
            return np.float16
        if value.dtype == torch.float32:
            return np.float32
        if value.dtype == torch.float64:
            return np.float64
        return np.float32
    return np.asarray(value).dtype


def _to_torch_float32(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)


def _to_numpy_float32(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def _semantic_shape(x, gamma):
    x_shape = _shape_of(x)
    gamma_rank = len(_shape_of(gamma))
    if gamma_rank == 0 or gamma_rank > len(x_shape):
        return x_shape
    return x_shape[: len(x_shape) - gamma_rank] + (1,) * gamma_rank


def _compute_mean_rstd(x32, gx32, rows, cols, semantic_shape, alpha32, epsilon):
    hidden = x32.reshape(rows, cols) * alpha32 + gx32.reshape(rows, cols)
    mean32 = torch.mean(hidden, dim=-1, keepdim=True)
    variance32 = torch.mean((hidden - mean32) * (hidden - mean32), dim=-1, keepdim=True)
    rstd32 = torch.rsqrt(variance32 + torch.tensor(float(epsilon), dtype=torch.float32))
    return mean32.reshape(semantic_shape), rstd32.reshape(semantic_shape)


def _get_power_split(count):
    power = VL_FP32
    while power <= count // 2:
        power *= 2
    return power


def _reduce_sum_vl(values):
    work = np.asarray(values, dtype=np.float32).reshape(-1)
    if work.size == 0:
        return np.float32(0.0)
    while work.size > 1:
        if work.size % 2 != 0:
            work = np.append(work, np.float32(0.0))
        work = (work[0::2] + work[1::2]).astype(np.float32)
    return work[0]


def _reduce_sum_regbase_1d(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    reduce_num = int(values.size)
    if reduce_num == 0:
        return np.float32(0.0)
    if reduce_num <= VL_FP32:
        return _reduce_sum_vl(values[:reduce_num])
    if reduce_num <= VL_FP32 + VL_FP32:
        folded = np.zeros((VL_FP32,), dtype=np.float32)
        folded += values[:VL_FP32]
        folded[: reduce_num - VL_FP32] += values[VL_FP32:reduce_num]
        return _reduce_sum_vl(folded)

    fold_point = _get_power_split(reduce_num)
    fold_loops = (fold_point + VL_FP32 - 1) // VL_FP32
    tail = reduce_num - fold_point
    tail_ceil_loops = (tail + VL_FP32 - 1) // VL_FP32
    tail_full_loops = tail // VL_FP32
    tmp = np.zeros((max(fold_loops, 2 * VL_FP32),), dtype=np.float32)
    for r in range(tail_full_loops):
        offset = r * VL_FP32
        summed = (
            values[offset : offset + VL_FP32]
            + values[fold_point + offset : fold_point + offset + VL_FP32]
        )
        tmp[r] = _reduce_sum_vl(summed)
    tail_remain = tail - tail_full_loops * VL_FP32
    if tail_remain != 0:
        offset = tail_full_loops * VL_FP32
        folded = np.array(
            values[offset : offset + VL_FP32], dtype=np.float32, copy=True
        )
        folded[:tail_remain] += values[
            fold_point + offset : fold_point + offset + tail_remain
        ]
        tmp[tail_full_loops] = _reduce_sum_vl(folded)
    for r in range(fold_loops - tail_ceil_loops):
        offset = tail_ceil_loops + r
        tmp[offset] = _reduce_sum_vl(values[offset * VL_FP32 : (offset + 1) * VL_FP32])

    if reduce_num <= VL_FP32 * VL_FP32 * DICHOTOMY_ADD_COEFF:
        return _reduce_sum_vl(tmp[: fold_point // VL_FP32])

    last_num = fold_point // VL_FP32 - VL_FP32
    folded = np.array(tmp[:VL_FP32], dtype=np.float32, copy=True)
    folded[:last_num] += tmp[VL_FP32 : VL_FP32 + last_num]
    return _reduce_sum_vl(folded)


def _reduce_sum_regbase(values, axis):
    values = np.asarray(values, dtype=np.float32)
    axis = axis % values.ndim
    reduce_num = values.shape[axis]
    if reduce_num <= MAX_TILE_LENGTH:
        return np.apply_along_axis(_reduce_sum_regbase_1d, axis, values).astype(
            np.float32
        )

    moved = np.moveaxis(values, axis, -1)
    result = np.zeros(moved.shape[:-1], dtype=np.float32)
    for start in range(0, reduce_num, MAX_TILE_LENGTH):
        tile = moved[..., start : start + MAX_TILE_LENGTH]
        tile_sum = np.apply_along_axis(_reduce_sum_regbase_1d, -1, tile).astype(
            np.float32
        )
        result = (result + tile_sum).astype(np.float32)
    return result


def _reduce_sum_rows(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros(values.shape[1:], dtype=np.float32)
    for row in values:
        result = (result + row).astype(np.float32)
    return result


def _is_row_constant(values):
    return np.all(values == values[:, :1], axis=1, keepdims=True)


def _dtype_name(dtype):
    text = str(dtype)
    if text in ("bfloat16", "ml_dtypes.bfloat16"):
        return "bfloat16"
    return np.dtype(dtype).name


def _fp32(x):
    return np.asarray(x).astype(np.float32)


def deep_norm_grad_input(
    dy, x, gx, gamma, mean, rstd, alpha=0.3, epsilon=1e-6, **kwargs
):
    """Generate forward-consistent mean and rstd while preserving TTK random inputs."""
    rows, cols = _flatten_shape(x, gamma)
    semantic_shape = _semantic_shape(x, gamma)
    if rows == 0 or cols == 0:
        zero = np.zeros(semantic_shape, dtype=np.float32)
        return [dy, x, gx, gamma, zero, zero]

    alpha32 = torch.tensor(float(alpha), dtype=torch.float32)
    x32 = _to_torch_float32(x)
    gx32 = _to_torch_float32(gx)
    mean32, rstd32 = _compute_mean_rstd(
        x32, gx32, rows, cols, semantic_shape, alpha32, epsilon
    )
    mean = _to_numpy_float32(mean32)
    rstd = _to_numpy_float32(rstd32)
    return [dy, x, gx, gamma, mean, rstd]


def deep_norm_grad_golden(dy, x, gx, gamma, mean, rstd, alpha=0.3, **kwargs):
    """Compute dx, dgx, dbeta and dgamma with device-aligned FP32 order."""
    rows, cols = _flatten_shape(x, gamma)
    x_shape = _shape_of(x)
    gx_shape = _shape_of(gx)
    gamma_shape = _shape_of(gamma)
    output_dtype = _numpy_dtype(x)
    if _numel(x) == 0 or cols == 0:
        return [
            np.zeros(x_shape, dtype=output_dtype),
            np.zeros(gx_shape, dtype=output_dtype),
            np.zeros(gamma_shape, dtype=np.float32),
            np.zeros(gamma_shape, dtype=np.float32),
        ]

    alpha32 = torch.tensor(float(alpha), dtype=torch.float32)
    inv_cols32 = np.float32(1.0 / float(cols))
    dy32 = _to_torch_float32(dy).reshape(rows, cols)
    x32 = _to_torch_float32(x).reshape(rows, cols)
    gx32 = _to_torch_float32(gx).reshape(rows, cols)
    gamma32 = _to_torch_float32(gamma).reshape(1, cols)
    mean32 = _to_torch_float32(mean)
    rstd32 = _to_torch_float32(rstd)
    semantic_shape = _semantic_shape(x, gamma)
    if mean32.numel() != rows or rstd32.numel() != rows:
        mean32, rstd32 = _compute_mean_rstd(
            x32, gx32, rows, cols, semantic_shape, alpha32, 1e-6
        )
    else:
        mean32 = mean32.reshape(rows, 1)
        rstd32 = rstd32.reshape(rows, 1)

    x_sum = x32 * alpha32 + gx32
    x_centered = x_sum - mean32
    tmp = dy32 * gamma32
    tmp_norm = tmp * rstd32
    product = rstd32 * rstd32
    product = product * rstd32
    product = product * x_centered
    product = product * tmp
    product_np = _to_numpy_float32(product)
    tmp_norm_np = _to_numpy_float32(tmp_norm)
    pd_var = _reduce_sum_regbase(product_np, axis=1).reshape(rows, 1) * np.float32(
        -inv_cols32
    )
    pd_mean = _reduce_sum_regbase(tmp_norm_np, axis=1).reshape(rows, 1) * np.float32(
        -inv_cols32
    )
    if _dtype_name(output_dtype) != "float16":
        pd_mean = np.where(
            _is_row_constant(tmp_norm_np), -tmp_norm_np[:, :1], pd_mean
        ).astype(np.float32)
    pd_var_t = torch.as_tensor(pd_var, dtype=torch.float32)
    pd_mean_t = torch.as_tensor(pd_mean, dtype=torch.float32)
    dgx = tmp_norm + x_centered * pd_var_t
    dgx = dgx + pd_mean_t
    dx = dgx * alpha32

    normalized = x_centered * rstd32
    dy_np = _to_numpy_float32(dy32)
    dbeta = _reduce_sum_rows(dy_np).reshape(gamma_shape)
    dgamma = _reduce_sum_rows(_to_numpy_float32(dy32 * normalized)).reshape(gamma_shape)

    return [
        _to_numpy_float32(dx.reshape(x_shape)).astype(output_dtype, copy=False),
        _to_numpy_float32(dgx.reshape(gx_shape)).astype(output_dtype, copy=False),
        dbeta.astype(np.float32, copy=False),
        dgamma.astype(np.float32, copy=False),
    ]
