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
"""TTK TestSpec golden for global_lp_pool (Kernel / GEIR path).

y = (sum_spatial |x|^p)^(1/p), keepdims=True, p != 0.
p edge behaviour: p == 0 or non-finite (+inf/-inf/NaN) -> ValueError;
empty reduction (prod(spatial) == 0) -> 0 (operator EMPTY_R convention).
数值口径：直接调用 torch.linalg.vector_norm，不做 dtype 提升或手写算法；
golden 与 third_party(torch) 同源，cross_check 退化为 NPU vs torch。
"""

import math

import ml_dtypes
import numpy as np
import torch

__spec__ = {
    "global_lp_pool": "GlobalLpPoolKernelSpec",
    "GlobalLpPool": "GlobalLpPoolKernelSpec",
}


def _np_to_torch(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr
    arr = np.ascontiguousarray(np.asarray(arr))
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return _torch_to_np(x)
    return np.asarray(x)


def _torch_to_np(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _norm_p(v):
    if isinstance(v, (np.ndarray, np.generic)):
        return float(v.item()) if v.ndim == 0 else float(v.reshape(-1)[0])
    if isinstance(v, torch.Tensor):
        return float(v.item()) if v.ndim == 0 else float(v.reshape(-1)[0])
    return float(v)


def _spatial_size(shape):
    n = 1
    for d in shape[2:]:
        n *= int(d)
    return n


def _out_shape(shape):
    return tuple(shape[:2]) + (1,) * (len(shape) - 2)


def _check_p(p_val):
    if p_val == 0.0 or not math.isfinite(p_val):
        raise ValueError(
            "global_lp_pool golden requires a finite non-zero p, got %r" % (p_val,)
        )


def _lp_norm(x, p_val):
    """函数形态 golden：numpy 输入 → numpy 输出；空归约按算子约定置 0。"""
    if _spatial_size(x.shape) == 0:
        return np.zeros(_out_shape(x.shape), dtype=x.dtype)
    xt = _np_to_torch(x)
    dims = tuple(range(2, xt.ndim))
    return _torch_to_np(
        torch.linalg.vector_norm(xt, ord=float(p_val), dim=dims, keepdim=True)
    )


def global_lp_pool_golden(x, p=2.0, **kwargs):
    p_val = _norm_p(p)
    _check_p(p_val)
    return [_lp_norm(_as_numpy(x), p_val)]


class GlobalLpPoolTorchSpec:
    def __init__(self, *, p=2.0, **kwargs):
        # 前置处理全部在此：p 归一化 + 合法性校验
        self.p = _norm_p(p)
        _check_p(self.p)

    def __call__(self, x, p=2.0, **kwargs):
        # x 已是 server 转好的 torch.Tensor（device 上）；空归约按算子约定置 0
        if _spatial_size(x.shape) == 0:
            return [x.new_zeros(_out_shape(x.shape))]
        return [
            torch.linalg.vector_norm(
                x, ord=self.p, dim=tuple(range(2, x.ndim)), keepdim=True
            )
        ]


class GlobalLpPoolKernelSpec:
    def golden(x, p=2.0, **kwargs):
        return global_lp_pool_golden(x, p, **kwargs)

    third_party = {"torch": GlobalLpPoolTorchSpec}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
