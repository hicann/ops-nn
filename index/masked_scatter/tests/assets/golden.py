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

__golden__ = {"kernel": {"masked_scatter": "masked_scatter_golden"}}


def masked_scatter_golden(input0, input1, input2, **kwargs):
    '''
    Golden function for masked_scatter.
    All the parameters (names and order) follow @masked_scatter_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    dtype = input0.dtype
    if "bfloat16" in str(dtype):
        input0 = input0.view("int16")
        input2 = input2.view("int16")

    x = torch.from_numpy(input0)
    tmpX = x.clone()
    mask = torch.from_numpy(input1)
    updates = torch.from_numpy(input2)
    res = tmpX.masked_scatter_(mask, updates).numpy()

    if "bfloat16" in str(dtype):
        res = res.view(dtype)
    return res
