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

__golden__ = {"kernel": {"relu_grad": "relu_grad_golden"}}


def relu_grad_golden(gradients, mask, **kwargs):
    '''
    Golden function for relu_grad.
    All the parameters (names and order) follow @relu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.ops import gen_nn_ops
    
    input0 = tf.compat.v1.placeholder(shape=gradients.shape, dtype=gradients.dtype.name)
    input1 = tf.compat.v1.placeholder(shape=mask.shape, dtype=mask.dtype.name)
    
    out = gen_nn_ops.relu_grad(input0, input1, name="relugrad")
    feed_dict = {input0: gradients, input1: mask}
    init_op = tf.compat.v1.global_variables_initializer()
    
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)
    return res
