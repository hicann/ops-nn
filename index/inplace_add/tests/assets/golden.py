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

The CPU golden is used only as the precision truth and is never timed as the XPU
competitor.  It computes in the dtype TTK supplies, so a Promote call that supplies
float32 or float64 is not cast back down to the original input dtype.  The
third-party leg may be used by current TTK for both cross-check precision and XPU
performance.  For non-empty work, both legs therefore express the operator with native
``torch.index_add`` and remain independent of the C++ kernel; the third-party leg
is deliberately kept as a real competitor implementation instead of being
replaced by a slower API merely to make its spelling differ from the golden.

Both legs share index normalization, wide-unsigned adaptation, and output
restoration.  In TTK's NumPy path, complex32 is represented by a trailing pair of
float16 components, so both legs add those components directly.  The explicit
Torch complex32 path keeps a compatibility promotion in the golden leg only.
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

# TTK cross_check currently supports only float16, bfloat16, and float32. The
# float16/float32 declarations select Promote plus the third-party leg and need a
# reachable XPU endpoint. BF16 uses binary_equal: this operator performs only one
# native-dtype addition per output element, while current cross_check calculates
# RMSE in float32 and overflows on valid full-range finite BF16 values. Complex
# outputs intentionally have no declaration here: resolve.py maps them to local
# isclose, or to binary_equal when the CLI explicitly selects `--compare binary`.
# Integer outputs always resolve to binary_equal.
#
# binary_equal first compares bytes, then treats output/golden NaNs at the same
# positions as equivalent if the byte comparison differs. Finite values and Inf
# remain byte-exact, while the NaN payload is outside the contract.
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
}

# Compatibility for callers that pass a real torch.complex32 Tensor directly.
# TTK's NumPy complex32 representation arrives here as float16 components and does
# not hit this mapping; both TTK reference legs therefore use the same component
# representation exercised by the binary kernel.
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


def _prepare_inplace_add_inputs(x, indices, v, *, copy_inputs=True):
    if isinstance(x, torch.Tensor):
        return_torch = True
        result_dtype = x.dtype
        y_source = x.clone() if copy_inputs else x
        updates_source = v.clone() if copy_inputs else v
        if result_dtype in (torch.uint16, torch.uint32, torch.uint64):
            y = y_source.to(torch.int64)
            updates = updates_source.to(torch.int64)
        else:
            y = y_source
            updates = updates_source
        idx_array = (
            indices
            if indices.dtype in (torch.int32, torch.int64)
            else indices.to(torch.int64)
        )
    else:
        return_torch = False
        x_array = np.array(x, copy=True) if copy_inputs else np.asarray(x)
        updates_array = np.array(v, copy=True) if copy_inputs else np.asarray(v)
        result_dtype = x_array.dtype
        if _is_wide_unsigned(result_dtype):
            y = torch.from_numpy(x_array.astype(np.int64))
            updates = torch.from_numpy(updates_array.astype(np.int64))
        else:
            y = _numpy_to_torch_tensor(x_array)
            updates = _numpy_to_torch_tensor(updates_array)
        indices_array = np.asarray(indices)
        if indices_array.dtype not in (np.dtype("int32"), np.dtype("int64")):
            indices_array = indices_array.astype(np.int64)
        idx_array = torch.from_numpy(indices_array)
    if y.shape[0] > 0:
        # torch.remainder already returns a non-negative result for positive N,
        # so one native operation exactly implements the mathematical modulo.
        idx_array = torch.remainder(idx_array, y.shape[0])
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
        x, indices, v, copy_inputs=False
    )
    if _is_empty_work(y, idx):
        return _restore_inplace_add_output(y.clone(), result_dtype, return_torch)
    # Keep the competitor leg in the native form measured by TTK's XPU path.
    # torch.index_add returns an independent output, so cloning x or v first would
    # add copies that are not part of the competitor implementation.
    updates = updates.reshape((idx.numel(),) + tuple(y.shape[1:]))
    y = torch.index_add(y, 0, idx.reshape(-1), updates)
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
# 【不存在】e2e 通路：op-plugin 517f2e7 中，aten::add_ 调用逐元素 aclnnInplaceAdd，
# 而 aten::index_add 调用 aclnnIndexAdd；两者都不会派发到本算子，当前没有可注册的
# indexed InplaceAdd torch API。
# 【不存在】ONNX 通路：framework/ 下只有 TensorFlow parser，无 ONNX parser。
# 注：TensorFlow parser 与 AInplaceAddFusionPass 是框架/图侧通路，按框架用例验证，
# 不作为 TestSpec 的 api_name 注册项。
# customize_inputs is not declared here: deterministic index generation stays in
# input.py under its __input__ registration, which the plugin loader falls back to.
