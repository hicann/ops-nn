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
        "aclnnInplaceIndexCopy": "aclnn_inplace_index_copy_golden",
    },
    "kernel": {"scatter_update": "scatter_update_golden"},
}


def scatter_update_golden(var, indices, updates, *, use_locking=False, **kwargs):
    """
    Golden function for scatter_update.
    All the parameters (names and order) follow @scatter_update_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()

    ref_tensor = tf.compat.v1.Variable(var, dtype=var.dtype)
    indices_tensor = tf.compat.v1.placeholder(indices.dtype, indices.shape)
    updates_tensor = tf.compat.v1.placeholder(updates.dtype, updates.shape)

    out = tf.compat.v1.scatter_update(
        ref_tensor, indices_tensor, updates_tensor, use_locking
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        res = sess.run(
            out, feed_dict={indices_tensor: indices, updates_tensor: updates}
        )

    return res


def aclnn_inplace_index_copy_golden(selfRef, dim, index, source, **kwargs):
    """
    Aclnn golden for aclnnInplaceIndexCopy.
    """
    import torch

    if hasattr(dim, "item"):
        dim = dim.item()

    if selfRef.dtype == torch.bfloat16:
        var = selfRef.to(torch.float32)
        updates = source.to(torch.float32)
    else:
        var = selfRef
        updates = source

    indices = index.to(torch.int64)
    result = torch.index_copy(var, dim=dim, index=indices, source=updates)
    return [result.to(selfRef.dtype)]
