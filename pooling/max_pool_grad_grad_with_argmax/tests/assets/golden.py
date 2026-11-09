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

"""MaxPoolGradGradWithArgmax Kernel/GEIR/TF TestSpec。"""

__spec__ = {
    "max_pool_grad_grad_with_argmax": "MaxPoolGradGradWithArgmaxKernelSpec",
    "tf.raw_ops.MaxPoolGradGradWithArgmax": "TfMaxPoolGradGradWithArgmaxSpec",
    "tf.compat.v1.raw_ops.MaxPoolGradGradWithArgmax": "TfMaxPoolGradGradWithArgmaxSpec",
}

import tensorflow as tf


def _gather_grad_by_argmax(
    grad,
    argmax,
    include_batch_in_index=False,
):
    """根据argmax索引搬运grad数据。"""
    if include_batch_in_index:
        grad_flat = tf.reshape(grad, [-1])
        argmax_flat = tf.reshape(argmax, [-1])

        output = tf.raw_ops.GatherV2(
            params=grad_flat,
            indices=argmax_flat,
            axis=tf.constant(0, dtype=tf.int32),
            batch_dims=0,
        )

        return tf.reshape(output, tf.shape(argmax))

    batch_size = tf.shape(grad)[0]
    grad_flat = tf.reshape(grad, [batch_size, -1])
    argmax_flat = tf.reshape(argmax, [batch_size, -1])

    output = tf.raw_ops.GatherV2(
        params=grad_flat,
        indices=argmax_flat,
        axis=tf.constant(1, dtype=tf.int32),
        batch_dims=1,
    )

    return tf.reshape(output, tf.shape(argmax))


def _tf_max_pool_grad_grad_with_argmax_tensor(
    input,
    grad,
    argmax,
    ksize,
    strides,
    padding,
    include_batch_in_index=False,
):
    """优先调用TensorFlow竞品接口，无可用GPU Kernel时使用等价语义兜底。"""
    ksize = [int(value) for value in ksize]
    strides = [int(value) for value in strides]
    padding = str(padding).upper()

    native_dtypes = {
        tf.float16,
        tf.float32,
        tf.float64,
    }

    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    use_native = has_gpu and grad.dtype in native_dtypes and argmax.dtype == tf.int64

    if use_native:
        with tf.device("/GPU:0"):
            return tf.raw_ops.MaxPoolGradGradWithArgmax(
                input=input,
                grad=grad,
                argmax=argmax,
                ksize=ksize,
                strides=strides,
                padding=padding,
                include_batch_in_index=include_batch_in_index,
            )

    return _gather_grad_by_argmax(
        grad,
        argmax,
        include_batch_in_index=include_batch_in_index,
    )


def _tf_max_pool_grad_grad_with_argmax(
    input,
    grad,
    argmax,
    ksize,
    strides,
    padding,
    include_batch_in_index=False,
):
    """同_tf_max_pool_grad_grad_with_argmax_tensor，输出numpy。"""
    return _tf_max_pool_grad_grad_with_argmax_tensor(
        input,
        grad,
        argmax,
        ksize,
        strides,
        padding,
        include_batch_in_index,
    ).numpy()


class MaxPoolGradGradWithArgmaxKernelSpec:
    """MaxPoolGradGradWithArgmax Kernel/GEIR TestSpec。"""

    def golden(
        x,
        grad,
        argmax,
        ksize,
        strides,
        padding,
        **kwargs,
    ):
        """使用TensorFlow小算子计算CPU Golden。"""
        grad_tensor = tf.convert_to_tensor(grad)
        argmax_tensor = tf.convert_to_tensor(argmax)

        output = _gather_grad_by_argmax(
            grad_tensor,
            argmax_tensor,
        )

        return [output.numpy()]

    class ThirdPartyImpl:
        """TensorFlow竞品实现。"""

        def __init__(
            self,
            x,
            grad,
            argmax,
            *,
            ksize,
            strides,
            padding,
            **kwargs,
        ):
            self.ksize = [int(value) for value in ksize]
            self.strides = [int(value) for value in strides]
            self.padding = str(padding).upper()

        def __call__(
            self,
            x,
            grad,
            argmax,
            **kwargs,
        ):
            output = _tf_max_pool_grad_grad_with_argmax_tensor(
                x,
                grad,
                argmax,
                self.ksize,
                self.strides,
                self.padding,
                False,
            )

            return [output]

    third_party = {
        "tf": ThirdPartyImpl,
    }

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint64": {"standard": "binary_equal"},
    }


class TfMaxPoolGradGradWithArgmaxSpec:
    """tf.raw_ops.MaxPoolGradGradWithArgmax E2E TestSpec。"""

    def golden(
        input,
        grad,
        argmax,
        ksize,
        strides,
        padding,
        include_batch_in_index=False,
        name=None,
        **kwargs,
    ):
        """E2E TF Golden。"""
        return [
            _tf_max_pool_grad_grad_with_argmax(
                input,
                grad,
                argmax,
                ksize,
                strides,
                padding,
                include_batch_in_index,
            )
        ]

    class ThirdPartyImpl:
        """TensorFlow竞品实现。"""

        def __init__(
            self,
            input,
            grad,
            argmax,
            ksize,
            strides,
            padding,
            include_batch_in_index=False,
            name=None,
            **kwargs,
        ):
            self.ksize = [int(value) for value in ksize]
            self.strides = [int(value) for value in strides]
            self.padding = str(padding).upper()
            self.include_batch_in_index = bool(include_batch_in_index)

        def __call__(
            self,
            input,
            grad,
            argmax,
            **kwargs,
        ):
            output = _tf_max_pool_grad_grad_with_argmax_tensor(
                input,
                grad,
                argmax,
                self.ksize,
                self.strides,
                self.padding,
                self.include_batch_in_index,
            )

            return [output]

    third_party = {
        "tf": ThirdPartyImpl,
    }

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint64": {"standard": "binary_equal"},
    }
