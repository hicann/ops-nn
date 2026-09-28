# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU oracle and native Torch reference for LpNormReduce.

Kernel/GEIR only. The oracle computes the reduction, not the final norm.
TTK promotes inputs for mix_tolerance/cross_check; retain that promoted dtype
on return instead of casting to the original device output dtype.
"""

import hashlib
import importlib
import operator

import numpy as np

__spec__ = {
    "lp_norm_reduce": "LpNormReduceTestSpec",
    "LpNormReduce": "LpNormReduceTestSpec",
}

P_INF = 2147483647
P_NEG_INF = -2147483648
INT64_MAX = (1 << 63) - 1


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    return operator.index(value)


def _parameters(shape, p, axes):
    """Validate metadata only; no shared numerical implementation with Torch."""
    p = _integer(p, "p")
    if p > INT64_MAX or (p < 0 and p != P_NEG_INF):
        raise ValueError(
            "p must be a nonnegative int64 or the negative-infinity sentinel"
        )
    rank = len(shape)
    if rank > 8:
        raise ValueError("rank must be in [0, 8]")
    if axes is None:
        axes = ()
    if not isinstance(axes, (list, tuple)):
        raise TypeError("axes must be a list or tuple of integers")
    dims = []
    for axis in axes:
        axis = _integer(axis, "axis")
        if axis < -rank or axis >= rank:
            raise ValueError("axis outside [-rank, rank)")
        axis %= rank
        if axis not in dims:
            dims.append(axis)
    dims = tuple(dims) if axes else tuple(range(rank))
    if p in (P_INF, P_NEG_INF) and any(shape[axis] == 0 for axis in dims):
        raise ValueError("infinite-p reduction over an empty domain is undefined")
    return p, dims


def _customize_inputs(x, **kwargs):
    """Inject deterministic special values by testcase name.

    The hook runs once inside TTK input generation, so NPU, golden and the
    remote third_party all consume the same injected arrays.

    - ``zeros_*``: about n/3 exact zeros (plus a few -0.0). The p=0 nonzero
      count semantics is only exercised when zeros actually exist.
    - ``infnan_*``: inject +inf/-inf/NaN/±0.0 for IEEE propagation checks.
    - Other cases pass through unchanged.

    Seed is the first 8 hex digits of md5(testcase_name) (builtin hash() is
    not stable across processes), keeping runs reproducible.
    """
    name = str(kwargs.get("testcase_name", ""))
    if not (name.startswith("zeros_") or name.startswith("infnan_")):
        return (x,)

    arr = np.asarray(x)
    if arr.size == 0:
        return (x,)
    seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1).copy()
    n = flat.size

    if name.startswith("zeros_"):
        idx = rng.choice(n, size=max(1, n // 3), replace=False)
        flat[idx] = 0.0
        # Keep a few -0.0: |-0.0| == 0 must not count as nonzero either.
        flat[idx[: max(1, idx.size // 10)]] = -0.0
    else:
        specials = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=flat.dtype)
        k = min(n, max(specials.size, n // 20))
        idx = rng.choice(n, size=k, replace=False)
        flat[idx] = specials[rng.integers(0, specials.size, size=k)]

    # ascontiguousarray promotes a scalar to (1,); restore its logical shape last.
    return (np.ascontiguousarray(flat).reshape(arr.shape),)


class TorchLpNormReduce:
    """Native Torch composition; FP16/BF16 intermediates use FP32 per spec.

    This implementation is also the performance reference. It does not mimic
    the device reduction tree, tiling or integer-power implementation.
    """

    def __init__(self, *, p=2, axes=None, keepdim=False, epsilon=1e-12, **kwargs):
        self.p = p
        self.axes = axes
        self.keepdim = keepdim

    def __call__(self, x, **kwargs):
        # Dynamic import: a literal ``import torch`` here makes TTK's worker
        # preload (pool.py:_preload_plugin_frameworks) import torch inside a
        # forkserver child, which segfaults in this environment. This class
        # only runs on the remote third_party server, which imports torch
        # itself; the local plugin never needs torch.
        torch = importlib.import_module("torch")

        p, dims = _parameters(x.shape, self.p, self.axes)
        calc = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        a = torch.abs(calc)
        if p == P_INF:
            y = torch.amax(a, dim=dims, keepdim=self.keepdim)
        elif p == P_NEG_INF:
            y = torch.amin(a, dim=dims, keepdim=self.keepdim)
        elif p == 0:
            y = (a != 0).to(calc.dtype).sum(dim=dims, keepdim=self.keepdim)
        elif p == 1:
            y = torch.sum(a, dim=dims, keepdim=self.keepdim)
        else:
            y = torch.sum(torch.pow(a, p), dim=dims, keepdim=self.keepdim)
        return [y.to(x.dtype)]


class LpNormReduceTestSpec:
    """CPU NumPy reference for the mathematical formula in the operator README."""

    @staticmethod
    def golden(x, p=2, axes=None, keepdim=False, epsilon=1e-12, **kwargs):
        """FP64 intermediates, input dtype on return (including Promote dtype).

        epsilon is intentionally unused. No clamping or added epsilon: preserve
        IEEE NaN/Inf behavior. NaN counts as nonzero for p=0. Scalar shape is
        retained; zero-length finite reductions return zero.
        """
        x = np.asarray(x)
        p, dims = _parameters(x.shape, p, axes)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            a = np.abs(x.astype(np.float64))
            if p == P_INF:
                y = np.max(a, axis=dims, keepdims=keepdim)
            elif p == P_NEG_INF:
                y = np.min(a, axis=dims, keepdims=keepdim)
            elif p == 0:
                y = np.sum(a != 0, axis=dims, keepdims=keepdim, dtype=np.float64)
            elif p == 1:
                y = np.sum(a, axis=dims, keepdims=keepdim)
            else:
                y = np.sum(np.power(a, p), axis=dims, keepdims=keepdim)
            return [np.asarray(y).astype(x.dtype)]

    third_party = {"torch": TorchLpNormReduce}
    customize_inputs = _customize_inputs
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
