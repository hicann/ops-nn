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

__golden__ = {"kernel": {"smooth_l1_loss_grad_v2": "smooth_l1_loss_grad_v2_golden"}}


mode_map = {
    "none": 0,
    "mean": 1,
    "sum": 2
}


def smooth_l1_loss_grad_v2_golden(predict, label, dout, *, sigma=1.0, reduction='mean', **kwargs):
    '''
    Golden function for smooth_l1_loss_grad_v2.
    All the parameters (names and order) follow @smooth_l1_loss_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    predict_shape = predict.shape
    label_shape = label.shape
    dout_shape = dout.shape

    max_shape = np.broadcast_shapes(predict_shape, label_shape, dout_shape)
    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)
    dout = np.broadcast_to(dout, max_shape)

    dtype_real = predict.dtype
    if 'bf16' in str(dtype_real) or 'bfloat16' in str(dtype_real):
        predicts = torch.from_numpy(predict.view(np.int16)).view(torch.bfloat16)
        target = torch.from_numpy(label.view(np.int16)).view(torch.bfloat16)
        grad_output = torch.from_numpy(dout.view(np.int16)).view(torch.bfloat16)
    else:
       predicts = torch.from_numpy(predict)
       target = torch.from_numpy(label)
       grad_output = torch.from_numpy(dout)

    ori_dtype = predicts.dtype

    if ori_dtype != torch.float32:
        predicts = predicts.to(torch.float32)
        target = target.to(torch.float32)
        grad_output = grad_output.to(torch.float32)

    reductionInt = mode_map.get(reduction, 1)
    res = torch.ops.aten.smooth_l1_loss_backward(grad_output, predicts, target, reductionInt, sigma)

    if ori_dtype != torch.float32:
        res = res.to(ori_dtype)

    if 'bf16' in str(dtype_real) or 'bfloat16' in str(dtype_real):
        return res.view(torch.int16).numpy().view(dtype_real)
    else:
        return res.numpy()
