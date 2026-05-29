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
import subprocess
import sys
import tempfile

__golden__ = {
    "kernel": {
        "apply_proximal_gradient_descent": "apply_proximal_gradient_descent_golden"
    }
}

_TF_SCRIPT = r'''
import sys, os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
data = np.load(sys.argv[1], allow_pickle=True)
var, alpha, l1, l2, delta = data["var"], data["alpha"], data["l1"], data["l2"], data["delta"]
var_tf = tf.Variable(var)
tf.raw_ops.ResourceApplyProximalGradientDescent(
    var=var_tf.handle,
    alpha=tf.constant(alpha.flatten()[0]),
    l1=tf.constant(l1.flatten()[0]),
    l2=tf.constant(l2.flatten()[0]),
    delta=tf.constant(delta)
)
np.save(sys.argv[2], var_tf.numpy())
'''


def apply_proximal_gradient_descent_golden(var, alpha, l1, l2, delta, **kwargs):
    '''
    Golden function for apply_proximal_gradient_descent.
    Delegates to tf.raw_ops.ResourceApplyProximalGradientDescent via subprocess.

    Args:
        var: weight tensor (numpy.ndarray)
        alpha: learning rate scalar tensor (numpy.ndarray)
        l1: L1 regularization coefficient scalar tensor (numpy.ndarray)
        l2: L2 regularization coefficient scalar tensor (numpy.ndarray)
        delta: gradient tensor (numpy.ndarray)

    Returns:
        Output tensor (numpy.ndarray)
    '''
    input_dtype = var.dtype
    if input_dtype.name in ("float16", "bfloat16"):
        var = var.astype("float32")
        alpha = alpha.astype("float32")
        l1 = l1.astype("float32")
        l2 = l2.astype("float32")
        delta = delta.astype("float32")

    with tempfile.NamedTemporaryFile(suffix=".npz") as fin, \
         tempfile.NamedTemporaryFile(suffix=".npy") as fout:
        np.savez(fin.name, var=var, alpha=alpha, l1=l1, l2=l2, delta=delta)
        proc = subprocess.run(
            [sys.executable, "-c", _TF_SCRIPT, fin.name, fout.name],
            capture_output=True, timeout=30
        )
        if proc.returncode != 0:
            raise RuntimeError(f"TF golden subprocess failed: {proc.stderr.decode()}")
        result = np.load(fout.name).astype(input_dtype, copy=False)

    return result
