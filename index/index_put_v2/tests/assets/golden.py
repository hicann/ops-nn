#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import copy
import numpy as np


__golden__ = {
    "kernel": {
        "index_put_v2": "index_put_v2_golden"
    }
}


def index_put_v2_golden(x, value, indexed_sizes, indexed_strides, indices, *, accumulate=False, **kwargs):
    '''
    Golden function for index_put_v2.
    All the parameters (names and order) follow @index_put_v2_def.cpp without outputs.
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
    else:
        x = copy.deepcopy(x)
    
    x_torch = torch.from_numpy(x)
    indices_list = [torch.from_numpy(arr.astype(np.int64)) for arr in indices]
    indices_tuple = tuple(indices_list)
    value_torch = torch.from_numpy(value)
    
    if accumulate:
        x_torch.index_put_(indices_tuple, value_torch, True)
    else:
        cmd = "x_torch["
        idx = 0
        for i in range(indexed_sizes.size):
            if indexed_sizes[i] > 0:
                cmd += "indices_tuple[{}]".format(idx)
                idx += 1
            else:
                cmd += ":"
            cmd += ","
        cmd += "] = value_torch"
        exec(cmd)
    
    if bf16_mark:
        x_torch = x_torch.to(torch.bfloat16)
    res = x_torch.numpy()
    return res
