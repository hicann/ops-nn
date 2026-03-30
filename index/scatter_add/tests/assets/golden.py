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

__golden__ = {"kernel": {"scatter_add": "scatter_add_golden"}}


def scatter_add_golden(var, indices, updates, *, use_locking=False, **kwargs):
    '''
    Golden function for scatter_add.
    All the parameters (names and order) follow @scatter_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    tensor_var = tf.Variable(var, dtype=var.dtype)
    indices_holder = tf.placeholder(indices.dtype, shape=indices.shape)
    updates_holder = tf.placeholder(updates.dtype, shape=updates.shape)
    
    out = tf.scatter_add(ref=tensor_var, indices=indices_holder, updates=updates_holder, use_locking=use_locking)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={indices_holder: indices, updates_holder: updates})
    return res
