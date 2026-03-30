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

__golden__ = {"kernel": {"lp_loss": "lp_loss_golden"}}


def lp_loss_golden(predict, label, *, reduction='mean', **kwargs):
    '''
    Golden function for lp_loss.
    All the parameters (names and order) follow @lp_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    predict_shape = predict.shape
    predict_dtype = predict.dtype

    has_improve_precision = False
    if "float16" in predict_dtype.name:
        predict = predict.astype("float32")
        label = label.astype("float32")
        has_improve_precision = True

    loss = np.abs(np.subtract(predict, label))

    if reduction == "sum":
        loss = np.sum(loss)
    elif reduction == "mean":
        reduce_elts = 1.0
        for i in predict_shape:
            reduce_elts *= i
        if reduce_elts == 0:
            loss = np.array([np.nan], dtype=predict_dtype)
            loss = loss.reshape([1])
        else:
            cof = reduce_elts ** (-1)
            div_loss = np.multiply(loss, cof)
            loss = np.sum(div_loss)
    elif reduction == "none":
        pass

    if has_improve_precision:
        loss = loss.astype(predict_dtype, copy=False)

    return loss
