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
    "kernel": {"scatter_nd_sub": "scatter_nd_sub_golden"}
}


def scatter_nd_sub_golden(var, indices, updates, *, use_locking=False, **kwargs):
    '''
    Golden function for scatter_nd_sub.
    All the parameters (names and order) follow @scatter_nd_sub_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Uses tf.tensor_scatter_nd_sub as golden baseline (TensorFlow competitor interface).
    Semantics: output = var - scatter_nd(indices, updates, var.shape)

    Args:
        var: numpy.ndarray, input tensor to be subtracted from, shape=[D0, D1, ..., Dn]
        indices: numpy.ndarray, index tensor, shape=[I0, I1, ..., Ik], dtype=int32/int64
        updates: numpy.ndarray, update tensor, shape=[I0, ..., Ik-1, DIk, ..., Dn]
        use_locking: bool, unused (TF tensor_scatter_nd_sub has no such param)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray), same shape/dtype as var
    '''
    var_tf = tf.constant(var)
    indices_tf = tf.constant(indices)
    updates_tf = tf.constant(updates)

    result_tf = tf.tensor_scatter_nd_sub(var_tf, indices_tf, updates_tf)

    return result_tf.numpy()
