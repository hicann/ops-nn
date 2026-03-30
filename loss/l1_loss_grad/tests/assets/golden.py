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

import functools
import numpy as np

__golden__ = {"kernel": {"l1_loss_grad": "l1_loss_grad_golden"}}


def l1_loss_grad_golden(grads, predict, label, *, reduction='mean', **kwargs):
    '''
    Golden function for l1_loss_grad.
    All the parameters (names and order) follow @l1_loss_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    input_dtype = grads.dtype
    if input_dtype.name == "bfloat16":
        grads = grads.astype("float32")
        label = label.astype("float32")
        predict = predict.astype("float32")
    sign = np.where(predict > label, 1.0, 0.0)
    sign = np.where(predict < label, -1.0, sign)
    label_shape = label.shape
    n = functools.reduce(lambda x, y: x * y, label_shape)
    if reduction == "mean":
        grads = np.divide(grads, n)
    return np.multiply(sign, grads).astype(input_dtype, copy=False)
