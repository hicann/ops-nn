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

__golden__ = {"kernel": {"leaky_relu_grad": "leaky_relu_grad_golden"}}


def leaky_relu_grad_golden(dy, y, *, negative_slope=0.0, **kwargs):
    '''
    Golden function for leaky_relu_grad.
    All the parameters (names and order) follow @leaky_relu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    if "bfloat16" in dy.dtype.name:
        dy_torch = torch.from_numpy(dy.view(np.int16)).view(torch.bfloat16)
        y_torch = torch.from_numpy(y.view(np.int16)).view(torch.bfloat16)
    else:
        dy_torch = torch.from_numpy(dy)
        y_torch = torch.from_numpy(y)

    result = torch.ops.aten.leaky_relu_backward(dy_torch, y_torch, negative_slope, False)
    if "bfloat16" in dy.dtype.name:
        return result.view(torch.int16).numpy().view(dy.dtype)
    return result.numpy()
