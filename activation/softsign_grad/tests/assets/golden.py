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

__golden__ = {
    "kernel": {
        "softsign_grad": "softsign_grad_golden"
    },
    "aclnn": {
        "aclnnSoftsignBackward": "aclnn_softsign_backward_golden"
    }
}


def softsign_grad_golden(gradients, features, **kwargs):
    '''
    Golden function for softsign_grad.
    Delegates to tf.raw_ops.SoftsignGrad.

    Args:
        gradients: upstream gradient tensor (numpy.ndarray)
        features: softsign forward input feature tensor (numpy.ndarray)

    Returns:
        Output tensor (numpy.ndarray)
    '''
    x_dtype = gradients.dtype
    with tf.device("/cpu:0"):
        result = tf.raw_ops.SoftsignGrad(
            gradients=tf.constant(gradients.astype(np.float32), dtype=tf.float32),
            features=tf.constant(features.astype(np.float32), dtype=tf.float32),
        )
    result = result.numpy()
    if x_dtype.name in ("bfloat16", "float16"):
        result = result.astype(x_dtype, copy=False)
    return result


def aclnn_softsign_backward_golden(gradients, features, out, **kwargs):
    '''
    Aclnn golden for aclnnSoftsignBackward.
    Delegates to tf.raw_ops.SoftsignGrad.

    Args:
        gradients: upstream gradient tensor (torch.Tensor)
        features: softsign forward input feature tensor (torch.Tensor)
        out: output tensor (pre-allocated, torch.Tensor)

    Returns:
        Output tensor (torch.Tensor)
    '''
    import torch

    origin_dtype = gradients.dtype
    grad_np = gradients.float().numpy()
    feat_np = features.float().numpy()

    with tf.device("/cpu:0"):
        result = tf.raw_ops.SoftsignGrad(
            gradients=tf.constant(grad_np, dtype=tf.float32),
            features=tf.constant(feat_np, dtype=tf.float32),
        )

    return torch.from_numpy(result.numpy()).to(origin_dtype)
