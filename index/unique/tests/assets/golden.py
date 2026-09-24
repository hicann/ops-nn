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

"""TTK CPU goldens for the public aclnnUnique APIs.

These are API-level references.  The direct values-only
``UniqueWithCountsAndSorting`` golden is maintained in that operator's test
directory instead.
"""

import numpy as np
import torch


__golden__ = {
    "aclnn": {
        "aclnnUnique": "aclnn_unique_golden",
        "aclnnUnique2": "aclnn_unique2_golden",
    }
}


def _as_bool(value):
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _is_aicpu_fallback(tensor):
    """Identify the dtypes using the ACLNN AICPU fallback in this package."""
    return tensor.dtype in (torch.bool, torch.float64)


def _stable_unique(tensor):
    """Return first-occurrence unique values, inverse indices, and counts."""
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        empty_index = torch.empty((0,), dtype=torch.int64, device=flat.device)
        return flat.clone(), empty_index, empty_index.clone()

    values = []
    inverse = []
    index_by_value = {}
    for item in flat.tolist():
        index = index_by_value.get(item)
        if index is None:
            index = len(values)
            index_by_value[item] = index
            values.append(item)
        inverse.append(index)
    unique_values = torch.tensor(values, dtype=flat.dtype, device=flat.device)
    inverse_tensor = torch.tensor(inverse, dtype=torch.int64, device=flat.device)
    counts = torch.bincount(inverse_tensor, minlength=unique_values.numel())
    return unique_values, inverse_tensor, counts


def _reference_unique(tensor, sorted_flag, return_inverse, return_counts):
    """Use the matching ACLNN route semantics for the CPU golden."""
    if _is_aicpu_fallback(tensor) and not sorted_flag:
        values, inverse, counts = _stable_unique(tensor)
        result = [values]
        if return_inverse:
            result.append(inverse)
        if return_counts:
            result.append(counts)
        return result

    # CPU torch.unique lacks parallel sort kernels for these unsigned dtypes.
    # NumPy preserves their full range; this ACLNN route sorts even when sorted=False.
    if tensor.dtype in (torch.uint16, torch.uint32, torch.uint64):
        values, inverse, counts = np.unique(
            tensor.detach().cpu().numpy().reshape(-1),
            return_inverse=True,
            return_counts=True,
        )
        result = [torch.from_numpy(values).to(tensor.device)]
        if return_inverse:
            result.append(
                torch.from_numpy(inverse.astype(np.int64))
                .reshape(tensor.shape)
                .to(tensor.device)
            )
        if return_counts:
            result.append(torch.from_numpy(counts.astype(np.int64)).to(tensor.device))
        return result

    result = torch.unique(
        tensor,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    if not isinstance(result, tuple):
        return [result]
    return list(result)


def _fit_output_capacity(values, output_tensor):
    """Match a capacity-shaped output buffer used by the ACLNN wrapper.

    ``countsOut`` keeps the input capacity as its ACL tensor view.  The kernel
    writes the valid prefix and leaves the remaining elements at TTK's
    pure-output initialization value (one).  PyTorch's reference returns only
    the valid unique prefix, so pad it before TTK's byte-wise comparison.
    """
    flat = values.reshape(-1)
    if output_tensor is None:
        return flat
    capacity = int(output_tensor.numel())
    if flat.numel() >= capacity:
        return flat[:capacity]
    padding = torch.ones(
        (capacity - flat.numel(),), dtype=flat.dtype, device=flat.device
    )
    return torch.cat((flat, padding), dim=0)


def aclnn_unique_golden(self, sorted, returnInverse, valueOut, inverseOut, **kwargs):
    """Reference aclnnUnique, omitting the disabled inverse output."""
    return_inverse = _as_bool(returnInverse)
    result = _reference_unique(self, _as_bool(sorted), return_inverse, False)
    if return_inverse:
        values, inverse = result
        return [values, inverse]
    return [result[0]]


def aclnn_unique2_golden(
    self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut, **kwargs
):
    """Reference aclnnUnique2, retaining only outputs enabled by its flags."""
    return_inverse = _as_bool(returnInverse)
    return_counts = _as_bool(returnCounts)
    result = _reference_unique(self, _as_bool(sorted), return_inverse, return_counts)
    if return_inverse and return_counts:
        values, inverse, counts = result
        return [
            values,
            inverse,
            counts
            if _is_aicpu_fallback(self)
            else _fit_output_capacity(counts, countsOut),
        ]
    if return_inverse:
        values, inverse = result
        return [values, inverse]
    if return_counts:
        values, counts = result
        return [
            values,
            counts
            if _is_aicpu_fallback(self)
            else _fit_output_capacity(counts, countsOut),
        ]
    return [result[0]]
