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

__golden__ = {"kernel": {"silu_grad": "silu_grad_golden"}}


def silu_grad_golden(dy, x, **kwargs):
    '''
    Golden function for silu_grad.
    All the parameters (names and order) follow @silu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    def _to_torch(arr):
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(arr)

    dy_torch = _to_torch(dy)
    x_torch = _to_torch(x)
    dx = torch.ops.aten.silu_backward(dy_torch, x_torch)

    if dx.dtype == torch.bfloat16:
        return dx.view(torch.int16).numpy().view(np.dtype('bfloat16'))
    else:
        return dx.numpy()
