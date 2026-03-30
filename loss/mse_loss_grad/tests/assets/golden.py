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

__golden__ = {"kernel": {"mse_loss_grad": "mse_loss_grad_golden"}}


def mse_loss_grad_golden(predict, label, dout, *, reduction="mean", **kwargs):
    '''
    Golden function for mse_loss_grad_golden.
    All the parameters (names and order) follow @mse_loss_grad_golden_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''

    ori_dtype = predict.dtype
    p_shape = predict.shape
    l_shape = label.shape
    d_shape = dout.shape

    max_shape = np.broadcast_shapes(p_shape, l_shape, d_shape)

    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)
    dout = np.broadcast_to(dout, max_shape)
    if ori_dtype == "float16" or ori_dtype == "bfloat16":
        predict = predict.astype("float32")
        label = label.astype("float32")
        dout = dout.astype("float32")
    sub_res = np.subtract(predict, label)
    if 'mean' == reduction:
        reduce_elts = 1.0
        for i in p_shape:
            reduce_elts *= i
        cof = (reduce_elts**(-1)) * 2.0
    else:
        cof = 2.0

    norm_grad = sub_res * cof
    grad_res = norm_grad * dout
    return grad_res.astype(ori_dtype, copy=False)
