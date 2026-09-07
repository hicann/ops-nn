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


__golden__ = {
    "aclnn": {
        "aclnnFlip": "aclnn_flip_golden",
    },
    "kernel": {"reverse_v2": "reverse_v2_golden"},
}


def reverse_v2_golden(x, axis, **kwargs):
    """
    Golden function for reverse_v2.
    All the parameters (names and order) follow @reverse_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    out = tf.reverse(x_holder, axis)
    with tf.Session() as sess:
        res = sess.run(out, feed_dict={x_holder: x})
    return res


def aclnn_flip_golden(self, dims=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnFlip.
    Parameters follow @aclnnFlipGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    if hasattr(dims, "item"):
        dims = [dims.item()]
    elif isinstance(dims, (list, tuple)):
        dims = [int(d) for d in dims]
    else:
        dims = [int(dims)]
    return [torch.flip(self, dims)]
