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

__golden__ = {"kernel": {"mish": "mish_golden"}}


def mish_golden(x, **kwargs):
    '''
    Golden function for mish.
    All the parameters (names and order) follow @mish_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    x_dtype = x.dtype
    if x_dtype.name == 'bfloat16':
        x_torch = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    else:
        x_torch = torch.from_numpy(x)
    
    if x_dtype.name == "bfloat16" or x_dtype.name == "float16":
        x_torch = x_torch.to(torch.float32)
    
    mish_func = torch.nn.Mish()
    output = mish_func(x_torch)
    
    return output.numpy().astype(x_dtype, copy=False)
