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

import numpy as np
import torch

__spec__ = {
    "inplace_sub": "InplaceSubKernelSpec",
}

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


def _prepare_updates(updates, idx, y_rank):
    if updates.ndim == idx.ndim and y_rank > 1:
        return updates.reshape(updates.shape + (1,) * (y_rank - 1))
    return updates


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


def _prepare_inplace_sub_inputs(x, indices, v):
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
        idx_array = torch.from_numpy(indices.astype(np.int64))
    if y.shape[0] > 0:
        idx_array = ((idx_array % y.shape[0]) + y.shape[0]) % y.shape[0]
    return y, idx_array, updates, result_dtype, return_torch


def _restore_inplace_sub_output(y, result_dtype, return_torch):
    if return_torch:
        return y.to(result_dtype)
    if _is_wide_unsigned(result_dtype):
        return y.numpy().astype(result_dtype)
    return _torch_to_numpy_tensor(y)


def _inplace_sub_golden_compute(x, indices, v):
    y, idx, updates, result_dtype, return_torch = _prepare_inplace_sub_inputs(
        x, indices, v
    )
    updates = _prepare_updates(updates, idx, y.dim())
    update_shape = (idx.numel(),) + tuple(y.shape[1:])
    updates = updates.expand(update_shape)
    y = torch.index_add(y, 0, idx.reshape(-1), -updates.reshape(update_shape))
    return _restore_inplace_sub_output(y, result_dtype, return_torch)


def _inplace_sub_third_party_compute(x, indices, v):
    y, idx, updates, result_dtype, return_torch = _prepare_inplace_sub_inputs(
        x, indices, v
    )
    updates = _prepare_updates(updates, idx, y.dim())
    y.index_put_((idx,), -updates, accumulate=True)
    return _restore_inplace_sub_output(y, result_dtype, return_torch)


class _InplaceSubCompose:
    def __call__(self, x, indices, v, **kwargs):
        return [_inplace_sub_third_party_compute(x, indices, v)]


class InplaceSubKernelSpec:
    @staticmethod
    def golden(x, indices, v, **kwargs):
        return [_inplace_sub_golden_compute(x, indices, v)]

    third_party = {"torch": _InplaceSubCompose}
    tolerance = _TOL


# Not registered in __spec__:
# - aclnn/e2e/ONNX: no public aclnn API, torch_npu binding, or ONNX parser is delivered.
# - TensorFlow parser and AInplaceSubFusionPass are validated as framework/graph routes,
#   not as TestSpec api_name entries.
