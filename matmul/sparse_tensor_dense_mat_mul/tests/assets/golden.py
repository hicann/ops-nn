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
__golden__ = {
        "kernel": {
            "sparse_tensor_dense_mat_mul": "sparse_tensor_dense_mat_mul_golden"
        }
}

import numpy as np

def sparse_tensor_dense_mat_mul_golden(x1_indices, x1_values, x1_shape, x2, adjoint_a: bool = False, adjoint_b: bool = False, **kwargs):
    try:
        import tensorflow as tf
    except ImportError:
        return _numpy_sparse_tensor_dense_matmul(x1_indices, x1_values, x1_shape, x2, adjoint_a, adjoint_b)
    
    x1_indices = tf.convert_to_tensor(x1_indices)
    x1_values = tf.convert_to_tensor(x1_values)
    x1_shape = tf.convert_to_tensor(x1_shape)
    x2 = tf.convert_to_tensor(x2)
    
    result = tf.raw_ops.SparseTensorDenseMatMul(
        a_indices=x1_indices,
        a_values=x1_values,
        a_shape=x1_shape,
        b=x2,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b
    )
    
    # 兼容 TF1.x 和 TF2.x
    y = None
    if tf.__version__.startswith("1."):
        with tf.Session() as sess:
            y =  sess.run(result)
    else:
        y = result.numpy()
    return y

def _numpy_sparse_tensor_dense_matmul(a_indices, a_values, a_dense_shape, b, adjoint_a, adjoint_b):
    """
    a_indices: (N, 2) int64
    a_values: (N,) float / double
    a_dense_shape: (2,) int64
    b: (K, N) or (N, K) dense numpy array
    adjoint_a / adjoint_b: 是否转置
    
    返回: numpy array
    """
    b = np.copy(b)
    
    # 构造稀疏矩阵
    A = np.zeros(a_dense_shape, dtype=a_values.dtype)
    for (i, j), v in zip(a_indices, a_values):
        A[i, j] = v

    if adjoint_a:
        A = A.T.conj()
    if adjoint_b:
        b = b.T.conj()

    return np.matmul(A, b)