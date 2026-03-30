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


__golden__ = {
    "kernel": {
        "apply_adagrad_d": "apply_adagrad_d_golden"
    }
}


def apply_adagrad_d_golden(var, accum, lr, grad, # inputs
                           update_slots: bool=True, use_locking: bool=False, # attributes
                           **kwargs):
    '''
    Golden function for apply_adagrad_d.
    All the parameters (names and order) follow @apply_adagrad_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    input_dtype = var.dtype
    if input_dtype.name in ("float16", "bfloat16"):
        var, accum = var.astype("float32"), accum.astype("float32")
        lr, grad = lr.astype("float32"), grad.astype("float32")
    
    res_accum = accum + grad * grad if update_slots else accum

    var_holder = tf.Variable(var)
    accum_holder = tf.Variable(accum)
    lr_holder = tf.constant(lr[0])
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    var_out = tf.raw_ops.ApplyAdagrad(var=var_holder, accum=accum_holder, lr=lr_holder, grad=grad_holder,
                                      use_locking=use_locking, update_slots=update_slots)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res_var = sess.run(var_out, feed_dict={grad_holder: grad})

    return [res_var.astype(input_dtype, copy=False),
            res_accum.astype(input_dtype, copy=False)]
