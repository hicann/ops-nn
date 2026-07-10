#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

__golden__ = {"kernel": {"sparse_segment_sum_grad": "sparse_segment_sum_grad_golden"}}

_NP_TO_TF_DTYPE = {
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "bfloat16": tf.bfloat16,
    "int32": tf.int32,
    "int64": tf.int64,
}


def _numpy_to_tf(arr):
    dtype_name = arr.dtype.name
    if dtype_name == "bfloat16":
        return tf.cast(tf.constant(arr.astype(np.float32)), tf.bfloat16)
    tf_dtype = _NP_TO_TF_DTYPE.get(dtype_name)
    if tf_dtype is None:
        return tf.constant(arr)
    return tf.cast(tf.constant(arr), tf_dtype)


def sparse_segment_sum_grad_golden(grad, indices, segment_ids, output_dim0, **kwargs):
    """
    Golden function for sparse_segment_sum_grad using TensorFlow SparseSegmentSumGrad.
    Equivalent to: output[indices[j]] += grad[segment_ids[j]]

    Args:
        grad: numpy array, shape (N, d1, d2, ...), output gradient
        indices: numpy array, shape (K,), index tensor pointing to original data rows
        segment_ids: numpy array, shape (K,), segment ID tensor pointing to grad rows
        output_dim0: numpy array, shape (1,), scalar tensor for original data first dim size
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy array), shape (output_dim0, d1, d2, ...)
    """
    orig_dtype = grad.dtype
    dim0_val = int(output_dim0.item())

    grad_t = _numpy_to_tf(grad)
    indices_t = tf.constant(indices.astype(np.int64), dtype=tf.int64)
    segment_ids_t = tf.constant(segment_ids.astype(np.int64), dtype=tf.int64)
    output_dim0_t = tf.constant(dim0_val, dtype=tf.int32)

    if len(indices) == 0:
        inner_shape = grad.shape[1:]
        result_np = np.zeros((dim0_val,) + inner_shape, dtype=orig_dtype)
    else:
        result_t = tf.raw_ops.SparseSegmentSumGrad(
            grad=grad_t,
            indices=indices_t,
            segment_ids=segment_ids_t,
            output_dim0=output_dim0_t,
        )
        if orig_dtype.name == "bfloat16":
            result_np = (
                tf.cast(result_t, tf.float32).numpy().astype(orig_dtype, copy=False)
            )
        else:
            result_np = result_t.numpy()

    return result_np
