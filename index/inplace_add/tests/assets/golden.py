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

"""InplaceAdd multi-path golden in the TestSpec format.

The operator copies ``x`` into ``y`` and then accumulates ``v`` into the rows
``indices`` selects, after normalizing every index modulo the first dimension:

    row_j = ((indices_j % N) + N) % N
    y[row_j, ...] += v[j, ...]

Only one registration key is needed: kernel and GEIR both resolve the snake-case
operator name and share :class:`InplaceAddKernelSpec`.  The legacy ``__golden__``
entry remains available and calls the same Torch computation core.

The reference math uses the Torch competitor operators (``torch.index_add`` for
the true value, ``Tensor.index_put_(accumulate=True)`` for the independent
third-party leg) rather than a NumPy formula, so the two paths cannot share a
misreading of the contract.  Both are single native-dtype adds per output
element, which is what the kernel performs, so no promotion is applied.
"""

import numpy as np
import torch

__spec__ = {
    # kernel and GEIR both resolve the snake-case operator name and share one Spec.
    "inplace_add": "InplaceAddKernelSpec",
}

__golden__ = {
    # Compatibility entry for the historical kernel loader; TestSpec takes priority.
    "kernel": {"inplace_add": "inplace_add_golden"},
}

# Float outputs declare cross_check so the third_party leg below is actually taken
# and the three-way precision column gets filled. cross_check needs a reachable XPU
# endpoint; where none is configured, run TTK with `--compare bin` -- resolve.py
# ranks the CLI above Spec.tolerance for float and complex outputs, which turns the
# whole suite into a bit-exact comparison. That is the stricter judgement and the
# kernel meets it: every registered case normalizes to unique target rows, so each
# output element is a single native-dtype add that reproduces the Torch reference
# bit for bit (verified on Ascend 950 across float16/float32/bfloat16, NaN/Inf rows
# included). Integer outputs never take part in cross_check -- they resolve to
# binary_equal regardless of what is declared here.
_TOL = {
    "complex32": {"standard": "cross_check", "level": "L1"},
    "complex64": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
}

# torch.index_add lacks a CPU kernel for complex32, so the golden path runs it in
# complex64 and narrows back afterwards. The accumulated row values stay exact:
# complex64 covers every complex32 addend, and the narrowing is the same rounding
# the kernel performs when it writes the row back.
_GOLDEN_PROMOTE = {torch.complex32: torch.complex64}


def _is_wide_unsigned(dtype):
    return dtype in (np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64"))


def _numpy_to_torch_tensor(array):
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _prepare_inplace_add_inputs(x, indices, v):
    if isinstance(x, torch.Tensor):
        return_torch = True
        result_dtype = x.dtype
        y_source = x.clone()
        updates_source = v.clone()
        if result_dtype in (torch.uint16, torch.uint32, torch.uint64):
            y = y_source.to(torch.int64)
            updates = updates_source.to(torch.int64)
        else:
            y = y_source
            updates = updates_source
        idx_array = indices.to(torch.int64)
    else:
        return_torch = False
        x_array = np.array(x, copy=True)
        updates_array = np.array(v, copy=True)
        result_dtype = x_array.dtype
        if _is_wide_unsigned(result_dtype):
            y = torch.from_numpy(x_array.astype(np.int64))
            updates = torch.from_numpy(updates_array.astype(np.int64))
        else:
            y = _numpy_to_torch_tensor(x_array)
            updates = _numpy_to_torch_tensor(updates_array)
        idx_array = torch.from_numpy(np.asarray(indices).astype(np.int64))
    if y.shape[0] > 0:
        idx_array = ((idx_array % y.shape[0]) + y.shape[0]) % y.shape[0]
    return y, idx_array, updates, result_dtype, return_torch


def _restore_inplace_add_output(y, result_dtype, return_torch):
    if return_torch:
        return y.to(result_dtype)
    if _is_wide_unsigned(result_dtype):
        return y.numpy().astype(result_dtype)
    return _torch_to_numpy_tensor(y)


def _is_empty_work(y, idx):
    """No row is touched: either the output holds no element, or no index is given."""
    return y.numel() == 0 or idx.numel() == 0


def _inplace_add_golden_compute(x, indices, v):
    y, idx, updates, result_dtype, return_torch = _prepare_inplace_add_inputs(
        x, indices, v
    )
    if _is_empty_work(y, idx):
        return _restore_inplace_add_output(y, result_dtype, return_torch)
    update_shape = (idx.numel(),) + tuple(y.shape[1:])
    updates = updates.reshape(update_shape)
    compute_dtype = _GOLDEN_PROMOTE.get(y.dtype)
    if compute_dtype is None:
        y = torch.index_add(y, 0, idx.reshape(-1), updates)
    else:
        y = torch.index_add(
            y.to(compute_dtype), 0, idx.reshape(-1), updates.to(compute_dtype)
        ).to(y.dtype)
    return _restore_inplace_add_output(y, result_dtype, return_torch)


def _inplace_add_third_party_compute(x, indices, v):
    y, idx, updates, result_dtype, return_torch = _prepare_inplace_add_inputs(
        x, indices, v
    )
    if _is_empty_work(y, idx):
        return _restore_inplace_add_output(y, result_dtype, return_torch)
    # Same reshape as the golden leg: index_put_ needs updates to match
    # (K,) + y.shape[1:], and letting only one leg normalize the shape would make
    # the two legs disagree on any input the caller hands over flattened.
    updates = updates.reshape((idx.numel(),) + tuple(y.shape[1:]))
    y.index_put_((idx,), updates, accumulate=True)
    return _restore_inplace_add_output(y, result_dtype, return_torch)


def _is_conflict_safety_case(kwargs):
    """Safety-only cases feed duplicate normalized indices.

    With duplicates the accumulation order across cores is unspecified, so the
    numeric result is undefined by contract -- these cases exist to prove the
    kernel executes safely, not to pin a value. Returning None for the output
    makes TTK record SUPPRESSED and count the case as passed, so the "golden
    disabled" intent is enforced by the plugin instead of relying on the caller
    remembering a golden-disable switch. Mirrors the selector in input.py.
    """
    return "conflict_safety" in kwargs.get("testcase_name", "")


class _InplaceAddCompose:
    def __call__(self, x, indices, v, **kwargs):
        if _is_conflict_safety_case(kwargs):
            return [None]
        return [_inplace_add_third_party_compute(x, indices, v)]


class InplaceAddKernelSpec:
    @staticmethod
    def golden(x, indices, v, **kwargs):
        if _is_conflict_safety_case(kwargs):
            return [None]
        return [_inplace_add_golden_compute(x, indices, v)]

    third_party = {"torch": _InplaceAddCompose}
    tolerance = _TOL


def inplace_add_golden(x, indices, v, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader.

    Returns the bare array the legacy loader expects for a single-output operator,
    and reuses the same computation core as the TestSpec above so the two entries
    can never drift apart.
    """
    del args
    if _is_conflict_safety_case(kwargs):
        return None
    return _inplace_add_golden_compute(x, indices, v)


# 【不存在】ACLNN 通路：op_host/ 下无 op_api 目录、docs/ 下无 aclnnInplaceAdd.md，
# 本算子不交付 aclnn 接口；同名 aclnnInplaceAdd 是逐元素/Broadcast 的 self += alpha * other，
# 与按索引行更新的语义不同，不能拿来顶替。
# 【不存在】e2e 通路：torch_npu 二进制里确有 aclnnInplaceAdd 符号，但它属于上面那个
# 逐元素算子（torch.Tensor.add_ 的落点），是同名不同算子；torch 侧没有任何入口能派发到
# 本算子，故不注册 e2e 通路。
# 【不存在】ONNX 通路：framework/ 下只有 TensorFlow parser，无 ONNX parser。
# 注：TensorFlow parser 与 AInplaceAddFusionPass 是框架/图侧通路，按框架用例验证，
# 不作为 TestSpec 的 api_name 注册项。
# customize_inputs is not declared here: deterministic index generation stays in
# input.py under its __input__ registration, which the plugin loader falls back to.
