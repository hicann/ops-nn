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

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

__spec__ = {"fused_mul_apply_keras_momentum": "FusedMulApplyKerasMomentumTestSpec"}

_L1 = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _np_to_tf_dtype(np_dtype):
    s = str(np_dtype)
    if "bfloat16" in s:
        return tf.bfloat16
    elif "float16" in s:
        return tf.float16
    elif "float64" in s or "double" in s:
        return tf.float64
    return tf.float32


def _scalar(arr, tf_dtype):
    return tf.constant(np.asarray(arr).flat[0], dtype=tf_dtype)


def _to_numpy(x):
    """Convert torch.Tensor (incl. GPU / requires-grad) to numpy; no-op for ndarray."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ThirdPartyImpl:
    """Third-party reference implementation (precision promotion aligned with NPU).

    Competitor: tf.raw_ops.ResourceApplyKerasMomentum.
    Precision policy mirrors the NPU kernel (README: FP16 inputs are promoted to
    FP32 inside the kernel, computed fully in FP32, then cast back to FP16 on
    output): ALL inputs are promoted to FP32 BEFORE computation (including the
    fused Mul grad=x1*x2), and results are cast back to the original dtype.
    """

    def __call__(
        self,
        var,
        accum,
        lr,
        x1,
        momentum,
        x2,
        use_locking=False,
        use_nesterov=False,
        **kwargs,
    ):
        var = _to_numpy(var)
        accum = _to_numpy(accum)
        lr = _to_numpy(lr)
        x1 = _to_numpy(x1)
        momentum = _to_numpy(momentum)
        x2 = _to_numpy(x2)
        input_dtype = var.dtype

        # 与 NPU kernel 一致：FP16 输入先升 FP32 再计算（全程 FP32），输出还原原 dtype
        if "float16" in str(input_dtype):
            var = var.astype(np.float32)
            accum = accum.astype(np.float32)
            lr = lr.astype(np.float32)
            x1 = x1.astype(np.float32)
            momentum = momentum.astype(np.float32)
            x2 = x2.astype(np.float32)
            tf_dtype = tf.float32
        else:
            tf_dtype = _np_to_tf_dtype(input_dtype)

        grad = x1 * float(np.asarray(x2).flat[0])
        var_v = tf.Variable(tf.convert_to_tensor(var, dtype=tf_dtype))
        accum_v = tf.Variable(tf.convert_to_tensor(accum, dtype=tf_dtype))
        tf.raw_ops.ResourceApplyKerasMomentum(
            var=var_v.handle,
            accum=accum_v.handle,
            lr=_scalar(lr, tf_dtype),
            grad=tf.convert_to_tensor(grad, dtype=tf_dtype),
            momentum=_scalar(momentum, tf_dtype),
            use_nesterov=use_nesterov,
        )
        return [
            var_v.numpy().astype(input_dtype, copy=False),
            accum_v.numpy().astype(input_dtype, copy=False),
        ]


class FusedMulApplyKerasMomentumTestSpec:
    """TestSpec for fused_mul_apply_keras_momentum (kernel/GEIR mode).

    Competitor: tf.raw_ops.ResourceApplyKerasMomentum.
    Fused Mul (grad=x1*x2) is applied before calling the competitor API.
    For FP16 inputs: ALL inputs promoted to FP32 BEFORE computation,
    results cast back to original dtype.
    """

    def golden(
        var,
        accum,
        lr,
        x1,
        momentum,
        x2,
        use_locking=False,
        use_nesterov=False,
        **kwargs,
    ):
        input_dtype = var.dtype

        if "float16" in str(input_dtype):
            var = var.astype(np.float32)
            accum = accum.astype(np.float32)
            lr = lr.astype(np.float32)
            x1 = x1.astype(np.float32)
            momentum = momentum.astype(np.float32)
            x2 = x2.astype(np.float32)

        tf_dtype = _np_to_tf_dtype(var.dtype)
        grad = x1 * float(np.asarray(x2).flat[0])

        var_v = tf.Variable(tf.convert_to_tensor(var, dtype=tf_dtype))
        accum_v = tf.Variable(tf.convert_to_tensor(accum, dtype=tf_dtype))
        tf.raw_ops.ResourceApplyKerasMomentum(
            var=var_v.handle,
            accum=accum_v.handle,
            lr=_scalar(lr, tf_dtype),
            grad=tf.convert_to_tensor(grad, dtype=tf_dtype),
            momentum=_scalar(momentum, tf_dtype),
            use_nesterov=use_nesterov,
        )

        return [
            var_v.numpy().astype(input_dtype, copy=False),
            accum_v.numpy().astype(input_dtype, copy=False),
        ]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = _L1
