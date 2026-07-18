#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

__golden__ = {"kernel": {"smooth_l1_loss_grad": "smooth_l1_loss_grad_golden"}}


def smooth_l1_loss_grad_golden(predict, label, dout, *, sigma=1.0, **kwargs):
    """
    Golden function for smooth_l1_loss_grad.
    All the parameters (names and order) follow @smooth_l1_loss_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    formula: gradient = clamp((predict - label) / sigma, -1, 1) * dout

    Args:
        predict: numpy.ndarray, prediction values
        label: numpy.ndarray, target values (same shape/dtype as predict)
        dout: numpy.ndarray, upstream gradient (same shape/dtype as predict)
        sigma: float, smoothing threshold (> 0, default 1.0)
        **kwargs: metadata from TTK

    Returns:
        Output tensor (gradient, same shape/dtype as predict)
    """
    input_dtype = predict.dtype
    if input_dtype.name == "bfloat16":
        predict = predict.astype("float32")
        label = label.astype("float32")
        dout = dout.astype("float32")

    diff = predict - label
    c = np.clip(diff, -sigma, sigma)
    s = c * (1.0 / sigma)
    gradient = s * dout

    return gradient.astype(input_dtype, copy=False)
