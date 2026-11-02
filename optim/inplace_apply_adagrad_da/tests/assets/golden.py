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


__golden__ = {"kernel": {"inplace_apply_adagrad_da": "inplace_apply_adagrad_da_golden"}}


def inplace_apply_adagrad_da_golden(
    var,
    gradient_accumulator,
    gradient_squared_accumulator,
    grad,
    lr,
    l1,
    l2,
    global_step,
    **kwargs,
):
    """
    Golden function for inplace_apply_adagrad_da.
    Delegates to tf.compat.v1.raw_ops.ApplyAdagradDA.

    Inputs follow the operator definition order:
        var, gradient_accumulator, gradient_squared_accumulator, grad,
        lr, l1, l2, global_step

    Returns:
        [var_out, ga_out, gsa_out] -- same dtype as inputs.
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    input_dtype = var.dtype
    var = var.astype("float32")
    gradient_accumulator = gradient_accumulator.astype("float32")
    gradient_squared_accumulator = gradient_squared_accumulator.astype("float32")
    grad = grad.astype("float32")

    lr_val = float(np.asarray(lr).flat[0])
    l1_val = float(np.asarray(l1).flat[0])
    l2_val = float(np.asarray(l2).flat[0])
    gs_val = int(np.asarray(global_step).flat[0])

    # 独立图: 默认图在长寿命 worker 进程中跨用例累积 Variable,
    # global_variables_initializer() 会重初始化全部历史变量(含大 shape 用例的
    # 数百 MB 初始化器), 使 golden 单次调用从 0.1s 劣化到百秒级。
    with tf.Graph().as_default():
        var_holder = tf.Variable(var)
        ga_holder = tf.Variable(gradient_accumulator)
        gsa_holder = tf.Variable(gradient_squared_accumulator)
        grad_holder = tf.placeholder(grad.dtype, shape=grad.shape)
        lr_holder = tf.constant(lr_val, dtype=tf.float32)
        l1_holder = tf.constant(l1_val, dtype=tf.float32)
        l2_holder = tf.constant(l2_val, dtype=tf.float32)
        gs_holder = tf.constant(gs_val, dtype=tf.int64)

        out_op = tf.raw_ops.ApplyAdagradDA(
            var=var_holder,
            gradient_accumulator=ga_holder,
            gradient_squared_accumulator=gsa_holder,
            grad=grad_holder,
            lr=lr_holder,
            l1=l1_holder,
            l2=l2_holder,
            global_step=gs_holder,
            use_locking=False,
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(out_op, feed_dict={grad_holder: grad})
            res_var = sess.run(var_holder)
            res_ga = sess.run(ga_holder)
            res_gsa = sess.run(gsa_holder)

    return [
        res_var.astype(input_dtype, copy=False),
        res_ga.astype(input_dtype, copy=False),
        res_gsa.astype(input_dtype, copy=False),
    ]


class InplaceApplyAdagradDASpec:
    golden = staticmethod(inplace_apply_adagrad_da_golden)

    class _TFThirdParty:
        """Third-party reference: TF ApplyAdagradDA on remote CPU (no GPU kernel for this op)."""

        def __call__(
            self,
            var,
            gradient_accumulator,
            gradient_squared_accumulator,
            grad,
            lr,
            l1,
            l2,
            global_step,
            **kwargs,
        ):
            import tensorflow as tf
            import numpy as np

            # 竞争者以算子 native dtype 履约输出契约（TF 混合精度惯例：fp32 计算、
            # native 存储）。若返回裸 fp32，cross_check 竞争比值中三方误差恒为 0，
            # 分母被 small_value 兜底后 NPU 合法的 fp16 舍入必然超比。
            input_dtype = np.asarray(var).dtype
            var_a = np.asarray(var).astype(np.float32)
            ga_a = np.asarray(gradient_accumulator).astype(np.float32)
            gsa_a = np.asarray(gradient_squared_accumulator).astype(np.float32)
            grad_a = np.asarray(grad).astype(np.float32)

            # ApplyAdagradDA 无 GPU kernel，强制 CPU 避免设备分配错误
            with tf.device("/CPU:0"):
                var_v = tf.Variable(tf.convert_to_tensor(var_a))
                ga_v = tf.Variable(tf.convert_to_tensor(ga_a))
                gsa_v = tf.Variable(tf.convert_to_tensor(gsa_a))
                tf.raw_ops.ResourceApplyAdagradDA(
                    var=var_v.handle,
                    gradient_accumulator=ga_v.handle,
                    gradient_squared_accumulator=gsa_v.handle,
                    grad=tf.convert_to_tensor(grad_a),
                    lr=tf.constant(np.float32(np.asarray(lr).flat[0])),
                    l1=tf.constant(np.float32(np.asarray(l1).flat[0])),
                    l2=tf.constant(np.float32(np.asarray(l2).flat[0])),
                    global_step=tf.constant(np.int64(np.asarray(global_step).flat[0])),
                    use_locking=False,
                )
                return [
                    var_v.numpy().astype(input_dtype, copy=False),
                    ga_v.numpy().astype(input_dtype, copy=False),
                    gsa_v.numpy().astype(input_dtype, copy=False),
                ]

    third_party = {"tf": _TFThirdParty}

    tolerance = {
        dtype: {"standard": "cross_check", "level": "L2"}
        for dtype in ("float16", "float32")
    }


__spec__ = {"inplace_apply_adagrad_da": "InplaceApplyAdagradDASpec"}
