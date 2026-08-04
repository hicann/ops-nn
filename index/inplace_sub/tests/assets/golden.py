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

__golden__ = {"kernel": {"inplace_sub": "inplace_sub_golden"}}


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


def inplace_sub_golden(x, indices, v, **kwargs):
    x_array = np.array(x, copy=True)
    updates_array = np.array(v, copy=True)
    result_dtype = x_array.dtype
    if _is_wide_unsigned(result_dtype):
        y = torch.from_numpy(x_array.astype(np.int64))
        updates = torch.from_numpy(updates_array.astype(np.int64))
    else:
        y = _numpy_to_torch_tensor(x_array)
        updates = _numpy_to_torch_tensor(updates_array)
    idx_array = indices.astype(np.int64)
    if x_array.shape[0] > 0:
        idx_array = ((idx_array % x_array.shape[0]) + x_array.shape[0]) % x_array.shape[
            0
        ]
    idx = torch.from_numpy(idx_array)
    updates = _prepare_updates(updates, idx, y.dim())
    y.index_put_((idx,), -updates, accumulate=True)
    if _is_wide_unsigned(result_dtype):
        return y.numpy().astype(result_dtype)
    return _torch_to_numpy_tensor(y)
