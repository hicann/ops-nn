#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""INTrainingReduceV2 multi-path golden in the TestSpec format.

The operator keeps N and C and reduces the spatial axes
(NHWC: H/W; NDHWC: D/H/W; otherwise dimension 2 onward):

    sum        = sum(x)
    square_sum = sum(x * x)

Only one registration key is needed: kernel and GEIR both resolve the snake-case
operator name and share :class:`INTrainingReduceV2KernelSpec`.  The legacy
``__golden__`` entry remains available and calls the same Torch computation core.

The reference math uses Torch operators.  Promote mode computes both statistics
in FP64 and preserves FP64 outputs.  TTK's FP16-to-FP32 input promotion alone is
insufficient for a high-precision reference to this operator's FP32 accumulation
and outputs.  Other modes follow the kernel's FP32 computation and output contract.
"""

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "in_training_reduce_v2": "INTrainingReduceV2KernelSpec",
}

__golden__ = {
    "kernel": {"in_training_reduce_v2": "in_training_reduce_v2_golden"},
}


_TOL = {
    # Cover registered input dtypes; both outputs use the FP32 criterion.
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _kernel_compute_input(x):
    """Apply the operator precision contract without lowering a diagnostic FP64 input."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.to(torch.float32)
    return x


def _input_format(kwargs, default="ND"):
    for key in ("input_ori_formats", "input_formats"):
        formats = kwargs.get(key)
        if formats:
            value = formats[0]
            while isinstance(value, (list, tuple)):
                value = value[0]
            return str(value).upper()
    return default


def _spatial_dims(rank, kwargs, default="ND"):
    layout = _input_format(kwargs, default)
    if layout in ("NHWC", "NDHWC"):
        expected_rank = 4 if layout == "NHWC" else 5
        if rank != expected_rank:
            raise ValueError(f"{layout} input must have rank {expected_rank}")
        return tuple(range(1, rank - 1))
    return tuple(range(2, rank))


def _compute(x, **kwargs):
    """Return ``[sum, square_sum]`` as Torch tensors in def.cpp output order.

    Promote mode widens the supplied input values before both squaring and
    reduction, including when TTK has only promoted FP16 input to FP32.
    In other modes only the kernel's FP16-to-FP32 computation cast is needed.
    """
    x_compute = (
        x.to(torch.float64)
        if kwargs.get("golden_mode") == "Promote"
        else _kernel_compute_input(x)
    )
    reduce_dims = _spatial_dims(x_compute.ndim, kwargs)

    if not reduce_dims:
        # A rank-2 ND input has no spatial dimension, so each (N, C) row has R=1.
        sum_out = torch.clone(x_compute)
        square_sum_out = torch.square(x_compute)
    else:
        sum_out = torch.sum(x_compute, dim=reduce_dims, keepdim=True)
        square_sum_out = torch.sum(
            torch.square(x_compute), dim=reduce_dims, keepdim=True
        )
    return [sum_out, square_sum_out]


def _to_numpy_inputs(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.ascontiguousarray(x))


def _output_dtype_names(kwargs):
    output_dtypes = kwargs.get("output_dtypes") or []
    names = []
    for dtype in output_dtypes:
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        names.append(str(dtype))
    return names


def _kernel_golden(x, **kwargs):
    """Kernel/GEIR container adapter: NumPy in, NumPy list out."""
    outputs = _compute(_to_numpy_inputs(x), **kwargs)
    output_dtypes = _output_dtype_names(kwargs)
    result = []
    for index, output in enumerate(outputs):
        array = output.detach().cpu().contiguous().numpy()
        if kwargs.get("golden_mode") != "Promote":
            dtype = output_dtypes[index] if index < len(output_dtypes) else "float32"
            array = array.astype(dtype, copy=False)
        result.append(np.ascontiguousarray(array))
    return result


class _INTrainingReduceV2Compose:
    """Independent GPU competitor composition aligned with the operator contract.

    Operation order follows the implementation, rather than replacing the two
    outputs with a variance/mean identity:

    1. ``LoadTensorForDtypeTIn`` promotes FP16 lanes to FP32;
    2. the sum path reduces the original values;
    3. the square-sum path performs ``Mul(x, x)`` first, then reduces it;
    4. both NPU outputs are stored as FP32.

    High-level Torch cannot prescribe the exact RegBase VL=64 lane-wise
    ``ReduceSum``/pairwise tree used by ``in_training_reduce_v2_ar_full_reduce.h``.
    This composition therefore matches the observable dtype and operation
    sequence, while the backend owns the internal reduction tree.
    """

    def __init__(self, **kwargs):
        self.input_format = _input_format(kwargs)

    def __call__(self, x, **kwargs):
        x_compute = _kernel_compute_input(x)

        if _input_format(kwargs, self.input_format) in ("NHWC", "NDHWC"):
            dims = _spatial_dims(x_compute.ndim, kwargs, self.input_format)
            sum_out = torch.sum(x_compute, dim=dims, keepdim=True)
            square_sum_out = torch.sum(
                torch.mul(x_compute, x_compute), dim=dims, keepdim=True
            )
        elif x_compute.ndim > 2:
            rows = torch.reshape(x_compute, (*x_compute.shape[:2], -1))
            # Kernel sum path: ReduceSum over the original x lanes.
            sum_out = torch.sum(rows, dim=-1, keepdim=True)
            # Kernel square-sum path: Mul first, then an independent ReduceSum.
            squared = torch.mul(rows, rows)
            square_sum_out = torch.sum(squared, dim=-1, keepdim=True)
            output_shape = (*x_compute.shape[:2], *(1,) * (x_compute.ndim - 2))
            sum_out = torch.reshape(sum_out, output_shape)
            square_sum_out = torch.reshape(square_sum_out, output_shape)
        else:
            sum_out = torch.clone(x_compute)
            square_sum_out = torch.mul(x_compute, x_compute)

        # NPU T_SUM is always float.  This cast aligns the competitor outputs;
        # it does not alter the Promote-aware CPU true-value path in _compute().
        return [
            sum_out.to(dtype=torch.float32),
            square_sum_out.to(dtype=torch.float32),
        ]


class INTrainingReduceV2KernelSpec:
    """Shared kernel/GEIR TestSpec; parameters follow def.cpp (``x`` only)."""

    def golden(x, **kwargs):
        return _kernel_golden(x, **kwargs)

    third_party = {"torch": _INTrainingReduceV2Compose}
    tolerance = _TOL


def in_training_reduce_v2_golden(x, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader."""
    del args
    legacy_kwargs = dict(kwargs)
    if not legacy_kwargs.get("output_dtypes"):
        # The old entry predates Promote/TestSpec and promises fixed FP32 outputs.
        legacy_kwargs["output_dtypes"] = [["float32"], ["float32"]]
    return tuple(_kernel_golden(x, **legacy_kwargs))


# 【不存在】ACLNN 通路：op_host/CMakeLists.txt declares ``ACLNNTYPE aclnn_exclude``;
# this operator has neither op_host/op_api nor an aclnn interface document.
# 【不存在】e2e 通路：the approved support surface is GE-only (README.md), and the
# repository contains neither a torch binding nor an ACLNN entry for this operator.
