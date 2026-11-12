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
"""TTK TestSpec golden for global_lp_pool (Kernel path).

Independent CPU reference implementation — does NOT import or reuse any kernel product.
Implements the GlobalLpPool formula per spec.yaml math_semantics + DESIGN.md §1.3:

    y = (sum(|x|^p over spatial dims))^(1/p), keepdims=True

  Spatial dims = dims[2:] (H,W for 4D NCHW; D0,D1,D2 for 5D NCD0D1D2).
  All intermediate computation done in FP32 for precision (per spec.yaml
  numerical_stability: intermediate_accumulation_in_float32).

Operator classification: computational (max_relative tolerance, not bitwise_equal).
  -> cross_check tolerance (NPU vs torch third_party reference).

Supported dtypes: float16, float32, bfloat16 (input/output same dtype).

Attributes:
  p: optional float, default 2.0 (L2 norm). p=1 -> L1 (Abs fast path),
     p=2 -> L2 (Mul fast path), general p -> Abs->Pow->Sum->Pow chain.

Tolerance:
  - Kernel (GlobalLpPoolKernelSpec): cross_check (NPU output vs torch
    third_party reference).
"""

__spec__ = {
    "global_lp_pool": "GlobalLpPoolKernelSpec",
    "GlobalLpPool": "GlobalLpPoolKernelSpec",
}

import math

import numpy as np
import torch
import ml_dtypes


def _np_to_torch(arr):
    """numpy -> torch, preserving bf16 bit-exactly via int16 bridge (no fp32 round-trip)."""
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    arr = np.asarray(arr)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(arr)


def _torch_to_np(t):
    """torch -> numpy, preserving bf16 bit-exactly via int16 bridge."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _norm_p(v):
    """Normalize p attribute (numpy scalar / tensor / python float) -> python float."""
    if isinstance(v, (np.ndarray, np.generic)):
        return float(v.item()) if v.ndim == 0 else float(v.reshape(-1)[0])
    if isinstance(v, torch.Tensor):
        return float(v.item()) if v.ndim == 0 else float(v.reshape(-1)[0])
    return float(v)


def _core_compute(x_t, p):
    """Core compute in torch (FP32 intermediate). Returns torch.Tensor.

    y = (sum(|x|^p over spatial dims))^(1/p), keepdims=True.

    Inputs:
        x_t: torch.Tensor (already converted, any float dtype)
        p:   python float, Lp norm exponent (p > 0)

    Computation order (per DESIGN.md §10.6/§10.10):
        1. Cast to FP32 (if not already)
        2. |x|^p  (torch.abs + torch.pow)
        3. ReduceSum over spatial dims (dims[2:]), keepdims=True
        4. ^(1/p)  (torch.pow)
        5. Cast back to original dtype
    """
    orig_dtype = x_t.dtype

    spatial_size = 1
    for d in x_t.shape[2:]:
        spatial_size *= int(d)
    if spatial_size == 0 and math.isfinite(p) and p != 0.0:
        out_shape = tuple(x_t.shape[:2]) + (1,) * (x_t.ndim - 2)
        return torch.zeros(out_shape, dtype=orig_dtype, device=x_t.device)

    x_f32 = x_t.to(torch.float32)

    spatial_dims = list(range(2, x_f32.ndim))

    powered = torch.pow(torch.abs(x_f32), p)
    summed = torch.sum(powered, dim=spatial_dims, keepdim=True)
    result = torch.pow(summed, 1.0 / p)

    return result.to(orig_dtype)


class GlobalLpPoolTorchSpec:
    """third_party: GPU PyTorch independent implementation (for cross_check).

    Parameter names match global_lp_pool_def.cpp (snake_case); server binds
    by named_values. Reuses _core_compute to ensure structural identity with
    golden (torch floating-point ops are consistent across CPU/GPU, cross_check
    metric measures NPU vs torch reference numerical deviation).
    """

    def __call__(self, x, p=2.0, **kwargs):
        p_val = _norm_p(p)
        return [_core_compute(x, p_val)]


class GlobalLpPoolKernelSpec:
    """Kernel / GEIR flow — golden receives numpy.ndarray, returns list[numpy.ndarray].

    Parameters follow global_lp_pool_def.cpp (without output y):
      input: x (REQUIRED, 4D NCHW or 5D NCD0D1D2, fp16/fp32/bf16)
      attr:  p (OPTIONAL float, default 2.0)

    p is optional with a default; no required attrs -> no ``*`` separator needed.
    """

    def golden(x, p=2.0, **kwargs):
        p_val = _norm_p(p)
        x_t = _np_to_torch(x)
        y_t = _core_compute(x_t, p_val)
        return [_torch_to_np(y_t)]

    third_party = {"torch": GlobalLpPoolTorchSpec}

    tolerance = {
        "float16": {"standard": "cross_check"},
        "float32": {"standard": "cross_check"},
        "bfloat16": {"standard": "cross_check"},
    }
