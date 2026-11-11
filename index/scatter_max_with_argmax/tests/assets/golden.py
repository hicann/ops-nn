#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Golden for scatter_max_with_argmax, mirroring the arch35 kernel semantics.

Kernel algorithm (arch35 simt):
  Phase 1: y = -inf, argmax = M (M = number of indices, sentinel "uncovered").
  Phase 2: iterate indices in ascending i order; a position p is updated with
           update[i] when (a) p not covered yet and update is not NaN (first
           valid cover, even if value <= sentinel, e.g. -inf), or (b) update[i]
           is strictly greater than the current y (GT compare keeps first index
           on ties). NaN is ignored.
  Phase 3: positions still argmax == M (never covered by a non-NaN update) get
           y = 0.

The reference is implemented with a vectorized torch.scatter_reduce (amax then
amin over matched indices), which is provably equivalent to the kernel:
  - y[p]      = max over non-NaN updates targeting p (or 0 if none).
  - argmax[p] = smallest index i among non-NaN updates with value == y[p]
                (or M if none).
This vectorized form avoids the O(M) python loop of a naive per-index update and
runs identically on CPU (golden) and remote device tensors (third_party).
"""

import torch

__golden__ = {
    "kernel": {"scatter_max_with_argmax": "scatter_max_with_argmax_golden"},
}
__spec__ = {
    "scatter_max_with_argmax": "ScatterMaxWithArgmaxSpec",
}


def _numel(t):
    """Element count for torch.Tensor or numpy.ndarray."""
    return t.numel() if torch.is_tensor(t) else t.size


def _validate(x, indices, updates):
    """Contract checks (README: rank 1-8, indices 1D, 0 <= indices < N)."""
    if not 1 <= x.ndim <= 8:
        raise ValueError(f"x must have rank 1-8, got {tuple(x.shape)}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {tuple(indices.shape)}")
    expected_upd = (indices.shape[0],) + tuple(x.shape[1:])
    if tuple(updates.shape) != expected_upd:
        raise ValueError(
            f"updates must have shape {expected_upd}, got {tuple(updates.shape)}"
        )
    if _numel(indices) and (indices.min() < 0 or indices.max() >= x.shape[0]):
        raise ValueError("indices values must be in [0, x.shape[0])")


def _scatter_max_with_argmax_tensor(x, indices, updates):
    """Device-agnostic core; returns (y, argmax) torch tensors on x's device."""
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    indices_tensor = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
    updates_tensor = torch.as_tensor(updates, dtype=torch.float32)
    dev = x_tensor.device

    out_shape = tuple(x_tensor.shape)
    n = int(x_tensor.shape[0]) if x_tensor.dim() > 0 else 1
    m = int(indices_tensor.numel())
    d = int(x_tensor.numel() // n) if n > 0 else 0

    y = torch.full((n, d), float("-inf"), dtype=torch.float32, device=dev)
    argmax = torch.full((n, d), m, dtype=torch.int32, device=dev)
    if n == 0 or d == 0 or m == 0:
        y = torch.where(argmax == m, torch.zeros((), dtype=y.dtype, device=dev), y)
        return y.reshape(out_shape), argmax.reshape(out_shape)

    updates_tensor = updates_tensor.reshape(m, d)
    rows = indices_tensor.unsqueeze(1).expand(-1, d)

    # NaN does not participate in max: replace with -inf so it never wins amax;
    # `matched` below uses the original values and excludes NaN, so a position
    # covered only by NaN stays uncovered (kernel-consistent).
    reduce_updates = torch.where(
        torch.isnan(updates_tensor),
        torch.full_like(updates_tensor, float("-inf")),
        updates_tensor,
    )
    y.scatter_reduce_(0, rows, reduce_updates, reduce="amax", include_self=True)

    # argmax = smallest index among non-NaN updates equal to the row max.
    max_per_update = y[indices_tensor]
    matched = (updates_tensor == max_per_update) & ~torch.isnan(updates_tensor)
    positions = (
        torch.arange(m, dtype=torch.int32, device=dev).unsqueeze(1).expand(-1, d)
    )
    candidate = torch.where(
        matched,
        positions,
        torch.full_like(matched, m, dtype=torch.int32),
    )
    argmax.scatter_reduce_(0, rows, candidate, reduce="amin", include_self=True)

    uncovered = argmax == m
    y = torch.where(uncovered, torch.zeros((), dtype=y.dtype, device=dev), y)
    return y.reshape(out_shape), argmax.reshape(out_shape)


def scatter_max_with_argmax_golden(x, indices, updates, **kwargs):
    """Legacy kernel/GEIR entry: numpy in -> (y, argmax) numpy."""
    del kwargs
    _validate(x, indices, updates)
    y, argmax = _scatter_max_with_argmax_tensor(x, indices, updates)
    return y.numpy(), argmax.numpy()


class ThirdPartyImpl:
    """GPU-side (torch) reference; runs the identical core on device tensors."""

    def __init__(self, x, indices, updates, device=None, **kwargs):
        del device, kwargs
        _validate(x, indices, updates)
        self.x = x
        self.indices = indices
        self.updates = updates

    def __call__(self, **kwargs):
        del kwargs
        y, argmax = _scatter_max_with_argmax_tensor(self.x, self.indices, self.updates)
        return [y, argmax]


class ScatterMaxWithArgmaxSpec:
    """golden (CPU numpy) and third_party (GPU tensor) share one implementation."""

    @staticmethod
    def golden(x, indices, updates, **kwargs):
        del kwargs
        _validate(x, indices, updates)
        y, argmax = _scatter_max_with_argmax_tensor(x, indices, updates)
        return y.numpy(), argmax.numpy()

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "int32": {"standard": "binary_equal"},
    }
