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

__golden__ = {
    "kernel": {"softplus": "softplus_golden"},
    "aclnn": {"aclnnSoftplus": "aclnn_softplus_golden"},
    "e2e": {"tf.nn.softplus": "softplus_e2e_golden"},
}


def softplus_golden(x, **kwargs):
    """
    Golden function for softplus.
    All the parameters (names and order) follow @softplus_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    ori_dtype = x.dtype
    if str(ori_dtype) == "bfloat16":
        x = x.astype("float32")

    input_placeholder = tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
    out = tf.nn.softplus(input_placeholder, name="softplus")
    feed_dict = {input_placeholder: x}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)

    return res.astype(ori_dtype, copy=False)


def aclnn_softplus_golden(self, beta=1, threshold=20, out=None, **kwargs):
    """
    Aclnn golden for aclnnSoftplus.
    Parameters follow @aclnnSoftplusGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.

    Formula:
        out = (1/beta) * log(1 + exp(beta * self))   when beta * self <= threshold
            = self                                     when beta * self > threshold
    """
    import torch

    orig_dtype = self.dtype
    x = self
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    result = torch.nn.functional.softplus(x, beta=beta, threshold=threshold)
    return [result.to(orig_dtype)]


def softplus_e2e_golden(features, **kwargs):
    """
    E2E golden for tf.nn.softplus.
    Computes plain softplus: y = log(1 + exp(x)).
    Input may be a torch.Tensor or numpy.ndarray.
    """
    import torch

    if isinstance(features, torch.Tensor):
        orig_dtype = features.dtype
        x = features
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        result = torch.nn.functional.softplus(x, beta=1, threshold=20)
        return [result.to(orig_dtype)]

    x = np.asarray(features)
    orig_dtype = x.dtype
    if x.dtype.name in ("float16", "bfloat16"):
        x = x.astype(np.float32)
    x_torch = torch.from_numpy(x)
    result = torch.nn.functional.softplus(x_torch, beta=1, threshold=20)
    return [result.numpy().astype(orig_dtype, copy=False)]
