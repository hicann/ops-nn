#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable NanMedian Kernel/GEIR reference and PyTorch benchmark.

Values use native ``torch.nanmedian`` where supported.  Its tie index is not a
stable contract, so indices use stable PyTorch small operators.  ACLNN/E2E/ONNX
are not delivered by this module's CMake target.
"""

import torch

__spec__ = {"nan_median": "NanMedianKernelSpec"}
__golden__ = {"kernel": {"nan_median": "nan_median_golden"}}
__input__ = {"kernel": {"nan_median": "nan_median_input"}}

_KERNEL_TOLERANCE = {
    dtype: {
        "standard": "stat_rel_err"
        if dtype in ("float16", "float32", "bfloat16")
        else "binary_equal"
    }
    for dtype in (
        "float16",
        "float32",
        "bfloat16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
}

_NATIVE_UNSUPPORTED_DTYPES = (torch.uint16, torch.uint32, torch.uint64)


def _from_array(x):
    if "bfloat16" in str(x.dtype):
        return torch.from_numpy(x.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(x)


def _to_array(x):
    x = x.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return x.view(torch.int16).numpy().view(bfloat16)
    return x.numpy()


def _stable_median(x, dim):
    axis = int(dim)
    rank = (x.shape[axis] - 1) // 2
    values, indices = torch.sort(x, dim=axis, stable=True)
    return values.narrow(axis, rank, 1), indices.narrow(axis, rank, 1)


def _stable_nan_median(x, dim):
    axis = int(dim)
    if not x.is_floating_point():
        selected_value, selected_index = _stable_median(x, axis)
        if x.dtype not in _NATIVE_UNSUPPORTED_DTYPES:
            selected_value = torch.nanmedian(x, dim=axis, keepdim=True).values
        return selected_value, selected_index
    values, indices = torch.sort(x, dim=axis, stable=True)
    valid_count = torch.sum(~torch.isnan(x), dim=axis, keepdim=True)
    rank = torch.clamp((valid_count - 1) // 2, min=0)
    selected_value = torch.gather(values, axis, rank)
    selected_index = torch.gather(indices, axis, rank)
    native_value = torch.nanmedian(x, dim=axis, keepdim=True).values
    special = torch.isnan(native_value) | (native_value == 0)
    return torch.where(special, selected_value, native_value), selected_index


def _fill_pattern_along_axis(tensor, values, axis):
    """Repeat a pattern inside each logical row of the reduced axis."""
    axis = int(axis)
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"axis {axis} is out of bounds for rank {tensor.ndim}")
    pattern = torch.tensor(values, dtype=tensor.dtype)
    repeats = (tensor.numel() + pattern.numel() - 1) // pattern.numel()
    moved_shape = [tensor.shape[i] for i in range(tensor.ndim) if i != axis]
    moved_shape.append(tensor.shape[axis])
    data = pattern.repeat(repeats)[: tensor.numel()].reshape(moved_shape)
    tensor.copy_(data.movedim(-1, axis))


def nan_median_input(x, *, dim=-1, testcase_name="", **kwargs):
    name = str(testcase_name)
    patterns = {
        "duplicate": [2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0],
        "signed_zero": [-0.0, 0.0, 0.0, -0.0, 1.0, -1.0, 0.0, -0.0],
        "all_nan": [float("nan")] * 8,
        "nan": [3.0, float("nan"), 1.0, float("nan"), 2.0, 2.0, -1.0, 0.0],
    }
    for marker, values in patterns.items():
        if name.endswith(f"_{marker}"):
            tensor = _from_array(x)
            _fill_pattern_along_axis(tensor, values, dim)
            break
    return (x,)


def nan_median_golden(x, *, dim=-1, **kwargs):
    return [_to_array(y) for y in _stable_nan_median(_from_array(x), dim)]


class NanMedianThirdParty:
    """Timed native PyTorch API with fallback for unsupported unsigned types."""

    def __init__(self, *, dim=-1, **kwargs):
        self.dim = int(dim)

    def __call__(self, x, **kwargs):
        if x.dtype in _NATIVE_UNSUPPORTED_DTYPES:
            return list(_stable_nan_median(x, self.dim))
        return torch.nanmedian(x, dim=self.dim, keepdim=True)


class NanMedianKernelSpec:
    golden = staticmethod(nan_median_golden)
    customize_inputs = staticmethod(nan_median_input)
    third_party = {"torch": NanMedianThirdParty}
    tolerance = dict(_KERNEL_TOLERANCE)
