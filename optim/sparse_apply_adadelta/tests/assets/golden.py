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
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

__golden__ = {
    "kernel": {"sparse_apply_adadelta": "sparse_apply_adadelta_golden"}
}

_call_counter = 0


def sparse_apply_adadelta_golden(var, accum, accum_update, lr, rho, epsilon, grad, indices, *, use_locking=False, **kwargs):
    '''
    Golden function for sparse_apply_adadelta.
    Uses tf.raw_ops.SparseApplyAdadelta (TensorFlow competitor interface) via
    tf.compat.v1.Session graph mode, because this op requires Ref(T) mutable
    inputs which are not supported in TF 2.x eager execution.

    Args:
        var: numpy.ndarray, shape=(D0, D1, ...), dtype=float32
        accum: numpy.ndarray, shape=(D0, D1, ...), dtype=float32
        accum_update: numpy.ndarray, shape=(D0, D1, ...), dtype=float32
        lr: numpy.ndarray, shape=(1,), dtype=float32
        rho: numpy.ndarray, shape=(1,), dtype=float32
        epsilon: numpy.ndarray, shape=(1,), dtype=float32
        grad: numpy.ndarray, shape=(N, D1, ...), dtype=float32
        indices: numpy.ndarray, shape=(N,), dtype=int32/int64
        use_locking: bool
        **kwargs: TTK metadata

    Returns:
        tuple: (var_out, accum_out, accum_update_out)
    '''
    global _call_counter
    _call_counter += 1
    suffix = str(_call_counter)

    lr_val = float(lr.flatten()[0]) if isinstance(lr, np.ndarray) else float(lr)
    rho_val = float(rho.flatten()[0]) if isinstance(rho, np.ndarray) else float(rho)
    epsilon_val = float(epsilon.flatten()[0]) if isinstance(epsilon, np.ndarray) else float(epsilon)

    grad_f32 = grad.astype(np.float32)
    indices_arr = np.asarray(indices)

    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            var_ref = tf.compat.v1.get_variable('var_' + suffix, initializer=var.astype(np.float32), use_resource=False)
            accum_ref = tf.compat.v1.get_variable('accum_' + suffix, initializer=accum.astype(np.float32), use_resource=False)
            au_ref = tf.compat.v1.get_variable('au_' + suffix, initializer=accum_update.astype(np.float32), use_resource=False)
            sess.run(tf.compat.v1.global_variables_initializer())

            op = tf.raw_ops.SparseApplyAdadelta(
                var=var_ref,
                accum=accum_ref,
                accum_update=au_ref,
                lr=lr_val,
                rho=rho_val,
                epsilon=epsilon_val,
                grad=grad_f32,
                indices=indices_arr,
                use_locking=use_locking,
            )
            sess.run(op)

            var_out, accum_out, accum_update_out = sess.run([var_ref, accum_ref, au_ref])

    return var_out, accum_out, accum_update_out
