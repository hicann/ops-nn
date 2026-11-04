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
"""ScaleNZ layout and host storage access for QBMM test assets."""

import os
import sys

import numpy as np
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def is_nz(formats, index):
    return bool(
        formats and len(formats) > index and formats[index] in ("NZ", "FRACTAL_NZ")
    )


def scale_nz_to_nd(data, logical_shape):
    """Decode (..., N1, Kg, 16, 2), preserving the canonical Kg/N order."""
    shape = tuple(logical_shape)
    if len(shape) < 3 or shape[-1] != 2:
        raise ValueError(f"ScaleNZ logical shape must end in (Kg, N, 2): {shape}")
    kg, n, _ = shape[-3:]
    storage_shape = shape[:-3] + ((n + 15) // 16, kg, 16, 2)
    if data.shape != storage_shape:
        raise ValueError(
            f"ScaleNZ storage shape {data.shape}, expected {storage_shape}"
        )
    values = data.swapaxes(-4, -3).reshape(shape[:-3] + (kg, storage_shape[-4] * 16, 2))
    return np.ascontiguousarray(values[..., :n, :])


def storage_view(tensor):
    """Return a writable flat view of the backing storage, on its original device."""
    if isinstance(tensor, torch.Tensor):
        count = tensor.untyped_storage().nbytes() // tensor.element_size()
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            tensor.untyped_storage(), 0, (count,), (1,)
        )
    base = tensor
    while getattr(base, "base", None) is not None:
        base = base.base
    array = np.asarray(base)
    flat = array.ravel(order="K")
    if array.size and not np.shares_memory(flat, tensor):
        raise ValueError(
            "NZ storage must have a writable contiguous backing allocation"
        )
    return flat


def full_storage(tensor):
    """Read the full storage on CPU; the returned array may be a copy."""
    return _util.to_numpy(storage_view(tensor))[0]


def aclnn_mx_nz_to_nd(tensor, *, scale=False):
    """Decode only MX WeightNZ/ScaleNZ; other quantization modes keep their path."""
    shape = tuple(tensor.shape)
    dtype = _util.dtype_to_str(tensor.dtype)
    if scale:
        if dtype != "float8_e8m0":
            raise ValueError(f"MX ScaleNZ requires E8M0, got {dtype}")
        physical = shape[:-3] + ((shape[-2] + 15) // 16, shape[-3], 16, 2)
        return scale_nz_to_nd(full_storage(tensor).reshape(physical), shape)
    c0_by_dtype = {
        "float4_e2m1": 64,
        "float4_e1m2": 64,
        "float8_e4m3fn": 32,
        "float8_e5m2": 32,
    }
    if dtype not in c0_by_dtype:
        raise ValueError(f"MX WeightNZ helper does not support {dtype}")
    c0 = c0_by_dtype[dtype]
    physical = shape[:-2] + ((shape[-1] + c0 - 1) // c0, (shape[-2] + 15) // 16, 16, c0)
    return _util.nz_to_nd(full_storage(tensor).reshape(physical), shape)
