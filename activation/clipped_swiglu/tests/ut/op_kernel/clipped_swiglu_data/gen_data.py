#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import torch
import tensorflow as tf

def do_clippedSwiglu(x, group_index, dim, alpha, limit, bias, interleaved):
    shape = x.shape
    x_dtype = x.dtype
    first_dim = torch.prod(torch.tensor(x.shape[:dim])).item()
    second_dim = torch.prod(torch.tensor(x.shape[dim:])).item()
    x = x.reshape(first_dim, second_dim)

    group = x.shape[0]
    if group_index is not None:
        group = min(group_index.sum(), x.shape[0])
    
    x_tensor = x[:group]
    if interleaved:
        x_glu = x_tensor[..., ::2]
        x_linear = x_tensor[..., 1::2]
    else:
        out = torch.chunk(x_tensor, 2, dim=-1)
        x_glu = out[0]
        x_linear = out[1]

    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    sigmoid_part = torch.sigmoid(alpha * x_glu)
    result = x_glu * sigmoid_part * (x_linear + bias)
    result = result.to(x_dtype)
    y = torch.zeros((x.shape[0], x.shape[1] // 2), dtype=x_dtype)
    y[:group] = result
    shape = list(shape)
    shape[dim] = shape[dim] // 2
    res = y.reshape(shape)

    return res

params_info = {
    "test_case_bf16_shortH": {"x_shape": [3200, 5760], "x_dtype": torch.bfloat16},
    "test_case_fp16_shortH": {"x_shape": [3200, 5760], "x_dtype": torch.float16},
    "test_case_fp32_shortH": {"x_shape": [3200, 5760], "x_dtype": torch.float32},
    "test_case_bf16_longH": {"x_shape": [3200, 23040], "x_dtype": torch.bfloat16},
    "test_case_fp16_longH": {"x_shape": [3200, 23040], "x_dtype": torch.float16},
    "test_case_fp32_longH": {"x_shape": [3200, 23040], "x_dtype": torch.float32}
    }


def gen_golden_data_simple(params_list):
    # np.random.seed(1)
    x_shape = params_list["x_shape"]
    x_dtype = params_list["x_dtype"]
    data_x = torch.randn(x_shape, dtype=torch.float32)
    group_index = torch.randint(low=1, high=11, size=(10,))
    out_list = do_clippedSwiglu(data_x, group_index, -1, 1.702, 7.0, 1.0, True)
    
    data_x = data_x.cpu().numpy()
    if params_list["x_dtype"] == torch.bfloat16:
        data_x = data_x.astype(tf.bfloat16.as_numpy_dtype)
    elif params_list["x_dtype"] == torch.float16:
        data_x = data_x.astype(np.float16)
    elif params_list["x_dtype"] == torch.float32:
        data_x = data_x.astype(np.float32)
    data_x.tofile("./input_x.bin")
    group_index.cpu().numpy().tofile("./input_group_index.bin")





if __name__ == "__main__":
    params_list = params_info[sys.argv[1]]
    gen_golden_data_simple(params_list)