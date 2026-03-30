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
__input__ = {
        "kernel": {
            "sparse_tensor_dense_mat_mul": "sparse_tensor_dense_mat_mul_inputs"
        }
}

import numpy as np
import random

def sparse_tensor_dense_mat_mul_inputs(x1_indices, x1_values, x1_shape, x2, adjoint_a: bool = False, adjoint_b: bool = False, **kwargs):
    ori_x1_indices = x1_indices
    x1_shape_list = x1_shape
    x1_shape = np.array(x1_shape_list, dtype=np.int64)

    nnz = x1_values.shape[0]
    assert nnz == ori_x1_indices.shape[0]
    [x1row, x1col] = x1_shape_list
    [x2row, x2col] = x2.shape

    [m, n] = [x1row, x1col] if not adjoint_a else [x1col, x1row]
    p = x2col if not adjoint_b else x2row
    if adjoint_b:
        assert n == x2col
    else:
        assert n == x2row

    # generate x1_indices
    x1_indices = generate_np_x1_indices(
        x1row, x1col, nnz, ori_x1_indices.dtype)

    return x1_indices, x1_values, x1_shape, x2

def generate_np_x1_indices(num_rows: int, num_cols: int, nnz: int, idx_dtype: np.dtype):
    if num_rows == 0 or num_cols == 0:
        return np.empty((0, 2), dtype=idx_dtype)
    all_idx = [(i, j) for i in range(num_rows) for j in range(num_cols)]
    return np.array(random.sample(all_idx, nnz), dtype=idx_dtype)