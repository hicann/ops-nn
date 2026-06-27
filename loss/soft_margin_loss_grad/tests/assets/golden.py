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

__golden__ = {"kernel": {"soft_margin_loss_grad": "soft_margin_loss_grad_golden"}}


def soft_margin_loss_grad_golden(grad_output, predict, label, *, reduction="mean", **kwargs):
    '''
    Golden function for soft_margin_loss_grad.
    Parameter names/order follow soft_margin_loss_grad_def.cpp (gradOutput, self, target)
    without outputs. All input Tensors are numpy.ndarray.

    grad_input = norm * (-target / (1 + exp(target * input))) * grad_output
               = norm * target * (sigmoid(target * input) - 1) * grad_output
    norm = 1/N (reduction=mean) else 1

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
    Returns:
        Output tensor with predict.dtype
    '''
    import torch

    ori_dtype = predict.dtype
    max_shape = np.broadcast_shapes(grad_output.shape, predict.shape, label.shape)
    grad_output = np.broadcast_to(grad_output, max_shape)
    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)

    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    reduction_int = reduction_map.get(reduction, 1)

    # bf16/fp16 走 fp32 影子计算，最后回写原 dtype
    cast = ori_dtype in ("float16", "bfloat16")
    grad_out_t = torch.from_numpy(grad_output.astype("float32") if cast else grad_output.copy())
    x_t = torch.from_numpy(predict.astype("float32") if cast else predict.copy())
    target_t = torch.from_numpy(label.astype("float32") if cast else label.copy())

    grad = torch.ops.aten.soft_margin_loss_backward(grad_out_t, x_t, target_t, reduction_int)
    return grad.numpy().astype(ori_dtype, copy=False)
