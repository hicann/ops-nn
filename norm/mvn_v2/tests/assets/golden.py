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
import tensorflow as tf

__spec__ = {"mvnv2": "MvnV2TestSpec"}


def mvn_v2_golden(x, eps, axes):
    """MVNV2 golden: y = (x - mean) / (sqrt(var) + eps)

    Args:
        x: input tensor (numpy array or TensorFlow tensor), 1-D to 8-D ND
        eps: numerical stability constant (float >= 0), added to std (not var)
        axes: reduction axes (list of int), default [0, 2, 3]

    Returns:
        y: output tensor, same shape and dtype as x
    """
    return_numpy = isinstance(x, np.ndarray)
    x_tensor = tf.convert_to_tensor(x)
    orig_dtype = x_tensor.dtype

    if axes is None:
        axes = [0, 2, 3]

    compute_dtype = tf.float64 if orig_dtype == tf.float32 else tf.float32
    x_compute = tf.cast(x_tensor, compute_dtype)
    mean = tf.reduce_mean(x_compute, axis=axes, keepdims=True)
    var = tf.reduce_mean(tf.square(x_compute - mean), axis=axes, keepdims=True)
    y_compute = (x_compute - mean) / (tf.sqrt(var) + tf.cast(eps, compute_dtype))

    y = tf.cast(y_compute, orig_dtype)
    return y.numpy() if return_numpy else y


def mvn_v2_tensorflow(x, eps=1.0e-9, axes=None, **kwargs):
    del kwargs
    if axes is None:
        axes = [axis for axis in (0, 2, 3) if axis < len(x.shape)]
    return [mvn_v2_golden(x, eps, axes)]


class MvnV2TestSpec:
    @staticmethod
    def golden(x, eps=1.0e-9, axes=None, **kwargs):
        del kwargs
        if axes is None:
            axes = [axis for axis in (0, 2, 3) if axis < x.ndim]
        return [mvn_v2_golden(x, eps, axes)]

    third_party = {"tf": mvn_v2_tensorflow}
    tolerance = {
        "float16": {"standard": "stat_rel_err", "threshold": 1.0e-3},
        "float32": {"standard": "stat_rel_err", "threshold": 1.0e-4},
    }
