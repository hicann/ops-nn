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

import numpy as np


__golden__ = {
    "kernel": {
        "apply_proximal_adagrad": "apply_proximal_adagrad_golden"
    }
}


def apply_proximal_adagrad_golden(var, accum, lr, l1, l2, grad, **kwargs):
    '''
    Golden function for apply_proximal_adagrad.
    Delegates to tf.raw_ops.ApplyProximalAdagrad.

    Inputs follow the operator definition order:
        var, accum, lr, l1, l2, grad     (numpy.ndarray each)

    Computation (per-element, all in fp32):
        accum_new = accum + grad * grad
        eta       = lr / sqrt(accum_new)
        prox      = var - eta * grad
        if l1 > 0:
            var_new = sign(prox) * max(|prox| - eta*l1, 0) / (1 + eta*l2)
        else:
            var_new = prox / (1 + eta*l2)

    Returns:
        [var_out, accum_out] -- same dtype as inputs (fp16/bf16 round-trip
        through fp32 to match the kernel cast convention).
    '''
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    input_dtype = var.dtype
    if input_dtype.name in ("float16", "bfloat16"):
        var = var.astype("float32")
        accum = accum.astype("float32")
        lr = lr.astype("float32")
        l1 = l1.astype("float32")
        l2 = l2.astype("float32")
        grad = grad.astype("float32")

    var_holder = tf.Variable(var)
    accum_holder = tf.Variable(accum)
    lr_holder = tf.constant(np.asarray(lr).reshape(()).astype("float32"))
    l1_holder = tf.constant(np.asarray(l1).reshape(()).astype("float32"))
    l2_holder = tf.constant(np.asarray(l2).reshape(()).astype("float32"))
    grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)

    var_out_op = tf.raw_ops.ApplyProximalAdagrad(
        var=var_holder, accum=accum_holder,
        lr=lr_holder, l1=l1_holder, l2=l2_holder,
        grad=grad_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res_var = sess.run(var_out_op, feed_dict={grad_holder: grad})
        res_accum = sess.run(accum_holder)

    return [res_var.astype(input_dtype, copy=False),
            res_accum.astype(input_dtype, copy=False)]
