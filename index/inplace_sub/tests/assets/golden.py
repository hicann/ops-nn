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
__golden__ = {"kernel": {"inplace_sub": "inplace_sub_golden"}}

_TOL = {
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


def _inplace_sub_compute(x, indices, v):
    if isinstance(x, torch.Tensor):
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
    idx = idx_array
    updates = _prepare_updates(updates, idx, y.dim())
    y.index_put_((idx,), -updates, accumulate=True)
    if isinstance(x, torch.Tensor):
        return y.to(result_dtype)
    if _is_wide_unsigned(result_dtype):
        return y.numpy().astype(result_dtype)
    return _torch_to_numpy_tensor(y)


class _InplaceSubCompose:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, indices, v, **kwargs):
        return [_inplace_sub_compute(x, indices, v)]


class InplaceSubKernelSpec:
    @staticmethod
    def golden(x, indices, v, **kwargs):
        return [_inplace_sub_compute(x, indices, v)]

    third_party = {"torch": _InplaceSubCompose}
    tolerance = _TOL


def inplace_sub_golden(x, indices, v, **kwargs):
    result_dtype = np.asarray(x).dtype
    y = _inplace_sub_compute(x, indices, v)
    if _is_wide_unsigned(result_dtype):
        return y.astype(result_dtype)
    return y


# Not registered in __spec__:
# - aclnn/e2e/ONNX: no public aclnn API, torch_npu binding, or ONNX parser is delivered.
# - TensorFlow parser and AInplaceSubFusionPass are validated as framework/graph routes,
#   not as TestSpec api_name entries.
