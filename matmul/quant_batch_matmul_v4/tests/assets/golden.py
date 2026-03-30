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
__golden__ = {
        "kernel": {
            "quant_batch_matmul_v4": "quant_batch_matmul_v4_golden"
        }
}

import numpy as np
import os
import torch
from ml_dtypes import bfloat16

def quant_batch_matmul_v4_golden(x1, x2, bias = None, x1_scale = None, x2_scale = None, y_scale = None,
                                 x1_offset = None, x2_offset = None, y_offset = None, x2_table = None,
                                 dtype:int = -1, compute_type:int = -1,
                                 transpose_x1:bool = False, transpose_x2:bool = False,
                                 group_size:int = -1, **kwargs):
    #获取需要的参数
    input_formats = kwargs['input_formats']
    x_dtype = x1.dtype
    output_dtypes = kwargs['output_dtypes']
    y_dtype = output_dtypes[0]
    weight_format = input_formats[1]
    testcase_name = kwargs['testcase_name']
    is_mx = False
    if (x2_scale is not None) and ("float8_e8m0" in x2_scale.dtype.name): #mxfp4
        is_mx = True
        x2 = x2.astype(np.float32)
    if weight_format == "FRACTAL_NZ":
        x2 = trans_nz2nd(x2)
    if transpose_x2:
        x2 = transpose_input(x2, transpose_x2)
        if len(x2_scale.shape) == 3:
            x2_scale = x2_scale.reshape(x2_scale.shape[0], -1)
        x2_scale = transpose_input(x2_scale, transpose_x2)
    if is_mx:
        if len(x1_scale.shape) == 3:
            x1_scale = x1_scale.reshape(x1_scale.shape[0], -1)
        k = x1.shape[-1]
        x1 = x1.astype(np.float32)

        for i in range(k):
            group_size_dim = i // group_size
            x1[:, i] = (x1[:, i]) * x1_scale[:, group_size_dim]
            x2[i, :] = (x2[i, :]) * x2_scale[group_size_dim, :]
        x1 = torch.from_numpy(x1.astype(np.float32))
        x2 = torch.from_numpy(x2.astype(np.float32))
        out = torch.matmul(x1, x2)
        if (bias is not None):
            out += bias
        out = out.numpy().astype(y_dtype)
    else:
        # Group场景，当前只处理GroupSize=32,此处需要把一个K广播为32个数据
        x2_scale_np_broadcast = np.repeat(x2_scale, group_size, axis=-2).astype(np.float32)
        # 模拟vcvt 输出是fp8:
        # Ascend950PR_9589是x2和x2_scale mul之后先降精度为bf16，再升精度为fp32，最后cast成fp8。原因为缺少bf16->fp8的指令
        x2_np = x2 * x2_scale_np_broadcast
        x2_np = x2_np.astype(bfloat16).astype(np.float32)
        x2_np = x2_np.astype(x_dtype)

        x2 = torch.from_numpy(x2_np.astype(np.float32))
        x1 = torch.from_numpy(x1.astype(np.float32))
        out = torch.matmul(x1, x2)
        #TTK不支持灌数据满足FP8*FP4的uint64数据，因此需要在inputs.py中做特殊处理
        fp32_deq_scale = np.load(testcase_name + "_deq_scale.npy")
        deq_scale_slice = fp32_deq_scale.reshape(1, -1)[:, :out.shape[-1]]

        out = (out * deq_scale_slice).numpy().astype(bfloat16)
        if os.path.exists(testcase_name + "_deq_scale.npy"):
            os.remove(testcase_name + "_deq_scale.npy")
    return out

# kn : n1 k1 k0 n0
# nk : k1 n1 n0 k0
def trans_nz2nd(input_data):
    nd_data = input_data.transpose(1, 2, 0, 3).reshape(input_data.shape[1] * input_data.shape[2], input_data.shape[0] * input_data.shape[3])
    return nd_data

def transpose_input(x, trans):
    if trans:
        array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0])
        return x.transpose(*array_trans)
    return x

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]
