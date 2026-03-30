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

__golden__ = {"kernel": {"softmax_grad": "softmax_grad_golden"}}


def softmax_grad_golden(softmax_data, grad_softmax_data, *, axes=-1, **kwargs):
    '''
    Golden function for softmax_grad.
    All the parameters (names and order) follow @softmax_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow as tf
    from tensorflow.python.ops import gen_math_ops
    
    axis = axes
    
    if softmax_data.dtype.name in ('float16', 'bfloat16'):
        softmax_data_fp32 = softmax_data.astype('float32')
        grad_softmax_data_fp32 = grad_softmax_data.astype('float32')
    else:
        softmax_data_fp32 = softmax_data
        grad_softmax_data_fp32 = grad_softmax_data
    
    tf.compat.v1.disable_eager_execution()
    softmax_shape = tf.compat.v1.placeholder(shape=softmax_data_fp32.shape, dtype=softmax_data_fp32.dtype)
    grad_softmax_shape = tf.compat.v1.placeholder(shape=grad_softmax_data_fp32.shape, dtype=grad_softmax_data_fp32.dtype)
    
    mul0 = gen_math_ops.mul(grad_softmax_shape, softmax_shape)
    sum0 = tf.reduce_sum(mul0, axis=axis, keepdims=True)
    mul1 = gen_math_ops.mul(sum0, softmax_shape)
    mul2 = gen_math_ops.mul(grad_softmax_shape, softmax_shape)
    sub0 = gen_math_ops.sub(mul2, mul1)
    
    feed_dict = {grad_softmax_shape: grad_softmax_data_fp32, softmax_shape: softmax_data_fp32}
    init_op = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(sub0, feed_dict=feed_dict)
    
    if softmax_data.dtype.name in ('float16', 'bfloat16'):
        res = res.astype(softmax_data.dtype.name)
    
    return res
