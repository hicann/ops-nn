#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
'''
gelu golden function
'''
import numpy as np

__golden__ = {
    "kernel": {
        "gelu": "gelu_golden"
    }
}


def gelu_golden(x, approximate="tanh", **kwargs):
    '''
    Golden function for gelu.
    All the parameters (names and order) follow @gelu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    input_dtype = x.dtype
    
    # Promote float16 and bfloat16 to float32 for computation
    if input_dtype.name == "float16" or input_dtype.name == "bfloat16":
        x = x.astype(np.float32)
    
    m = torch.nn.GELU(approximate=approximate)
    res = m(torch.from_numpy(x)).numpy()
    
    return res.astype(input_dtype, copy=False)
