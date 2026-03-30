#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np


__golden__ = {
    "kernel": {
        "index": "index_golden"
    }
}


def index_golden(x, mask, out, indices, **kwargs):
    '''
    Golden function for index.
    All the parameters (names and order) follow @index_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    bf16_mark = False
    if "bfloat16" in str(x.dtype):
        bf16_mark = True
        x = x.astype(np.float32)
    
    x_torch = torch.from_numpy(x)
    indices_list = [arr.astype(np.int64) for arr in indices]
    indices_torch = torch.from_numpy(np.array(indices_list))
    
    cmd = "x_torch["
    idx = 0
    for i in range(mask.size):
        if mask[i]:
            cmd += "indices_torch[{}]".format(idx)
            idx += 1
        else:
            cmd += ":"
        cmd += ","
    cmd += "]"
    res = eval(cmd)
    
    if bf16_mark:
        res.to(torch.bfloat16)
    res = res.numpy()
    return res
