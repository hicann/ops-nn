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
"""ApplyCenteredRMSProp 独立 Golden：Kernel / GEIR / TensorFlow。

The operator benchmarks against tf.raw_ops.ApplyCenteredRMSProp. Because
the Ref-based ``ApplyCenteredRMSProp`` does not support eager execution,
the equivalent ``ResourceApplyCenteredRMSProp`` is used — both share the
same kernel and compute identical results.

Reference precision is controlled by TTK Promote. All calculations retain
the supplied dtype, including FP64 inputs promoted by TTK cross_check.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf


__spec__ = {
    "apply_centered_rms_prop": "ApplyCenteredRMSPropKernelSpec",
    "tf.raw_ops.ApplyCenteredRMSProp": "ApplyCenteredRMSPropTensorFlowSpec",
}


_L1 = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _np_to_tf_dtype(np_dtype):
    return tf.as_dtype(np_dtype)


def _as_numpy(value):
    # TensorFlow BF16 scalars need .numpy() before NumPy conversion.
    return np.asarray(value.numpy() if tf.is_tensor(value) else value)


def _scalar(arr, tf_dtype):
    return tf.constant(float(_as_numpy(arr).flat[0]), dtype=tf_dtype)


def _compute(var, mg, ms, mom, lr, rho, momentum, epsilon, grad, use_locking=False):
    """Central computation using the dtype supplied by TTK Promote."""
    var = _as_numpy(var)
    output_dtype = var.dtype

    mg = _as_numpy(mg)
    ms = _as_numpy(ms)
    mom = _as_numpy(mom)
    grad = _as_numpy(grad)

    tf_dtype = _np_to_tf_dtype(var.dtype)
    var_v = tf.Variable(tf.convert_to_tensor(var, dtype=tf_dtype))
    mg_v = tf.Variable(tf.convert_to_tensor(mg, dtype=tf_dtype))
    ms_v = tf.Variable(tf.convert_to_tensor(ms, dtype=tf_dtype))
    mom_v = tf.Variable(tf.convert_to_tensor(mom, dtype=tf_dtype))
    tf.raw_ops.ResourceApplyCenteredRMSProp(
        var=var_v.handle,
        mg=mg_v.handle,
        ms=ms_v.handle,
        mom=mom_v.handle,
        lr=_scalar(lr, tf_dtype),
        rho=_scalar(rho, tf_dtype),
        momentum=_scalar(momentum, tf_dtype),
        epsilon=_scalar(epsilon, tf_dtype),
        grad=tf.convert_to_tensor(grad, dtype=tf_dtype),
        use_locking=use_locking,
    )

    return var_v.numpy().astype(output_dtype, copy=False)


def apply_centered_rms_prop_golden(
    var, mg, ms, mom, lr, rho, momentum, epsilon, grad, use_locking=False, **kwargs
):
    del kwargs
    with tf.device("/CPU:0"):
        return [
            _compute(
                var,
                mg,
                ms,
                mom,
                lr,
                rho,
                momentum,
                epsilon,
                grad,
                use_locking=use_locking,
            )
        ]


class _TensorFlowApplyCenteredRMSProp:
    """TensorFlow competitor: ``tf.raw_ops.ResourceApplyCenteredRMSProp``.

    Runs in the original input dtype without precision promotion.
    """

    def __call__(
        instance,
        var,
        mg,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        use_locking=False,
        **kwargs,
    ):
        del kwargs
        tf_dtype = _np_to_tf_dtype(var.dtype)
        var_v = tf.Variable(tf.convert_to_tensor(var, dtype=tf_dtype))
        mg_v = tf.Variable(tf.convert_to_tensor(mg, dtype=tf_dtype))
        ms_v = tf.Variable(tf.convert_to_tensor(ms, dtype=tf_dtype))
        mom_v = tf.Variable(tf.convert_to_tensor(mom, dtype=tf_dtype))
        tf.raw_ops.ResourceApplyCenteredRMSProp(
            var=var_v.handle,
            mg=mg_v.handle,
            ms=ms_v.handle,
            mom=mom_v.handle,
            lr=_scalar(lr, tf_dtype),
            rho=_scalar(rho, tf_dtype),
            momentum=_scalar(momentum, tf_dtype),
            epsilon=_scalar(epsilon, tf_dtype),
            grad=tf.convert_to_tensor(grad, dtype=tf_dtype),
            use_locking=use_locking,
        )
        return [var_v.numpy().astype(tf_dtype.as_numpy_dtype, copy=False)]


class ApplyCenteredRMSPropKernelSpec:
    """Kernel / GEIR spec registered by CSV op_name ``apply_centered_rms_prop``."""

    golden = staticmethod(apply_centered_rms_prop_golden)
    third_party = {"tf": _TensorFlowApplyCenteredRMSProp}
    tolerance = _L1


class ApplyCenteredRMSPropTensorFlowSpec:
    """TF ref-op spec; the third-party wrapper uses its eager resource equivalent."""

    golden = staticmethod(apply_centered_rms_prop_golden)
    third_party = {"tf": _TensorFlowApplyCenteredRMSProp}
    tolerance = _L1
