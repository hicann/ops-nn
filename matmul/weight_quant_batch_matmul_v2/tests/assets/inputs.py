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
__input__ = {
        "kernel": {
            "weight_quant_batch_matmul_v2": "weight_quant_batch_matmul_v2_inputs"
        }
}

import numpy as np
import math
import struct

def weight_quant_batch_matmul_v2_inputs(x1, weight, antiquant_scale, antiquant_offset = None, quant_scale =None, 
                                        quant_offset = None, bias =None, transpose_x: bool = False, 
                                        transpose_weight: bool = False, antiquant_group_size: int = 0,
                                        dtype: int = -1, inner_precise: int = 0,  **kwargs):
    output_dtypes = kwargs['output_dtypes']
    input_ranges = kwargs['input_ranges']
    testcase_name = kwargs['testcase_name']
    if antiquant_offset is not None:
        antiquant_offset = change_antiquant_dtype(antiquant_offset, x1.dtype, input_ranges[3])       # antiquant_offset 存在时修正数值范围

    if antiquant_scale.dtype.name == "uint64":
        antiquant_scale = weight_quant_bmmv2_uint64_antiquant_scale_generate(antiquant_scale, input_ranges, testcase_name)

    if output_dtypes[0] == "int8":
        quant_scale = weight_quant_bmmv2_uint64_quant_scale_generate(quant_scale.shape, testcase_name)

    return x1, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias
    
 # 伪量化算子量化参数dtype虽然是浮点,data_range要求是整型[-128,127]
def change_antiquant_dtype(antiquant_offset, x_dtype, input_data_range):
    input_range_left, input_range_right = input_data_range[0], input_data_range[1]
    input_range_left = math.ceil(input_range_left)
    input_range_right = math.ceil(input_range_right)
    if input_range_right > 127:
        input_range_right = 127
    elif input_range_left < -128: 
        input_range_left = -128
    if input_range_left == input_range_right:
        if x_dtype.name in ('float16', 'bfloat16'):
            antiquant_offset = np.full(antiquant_offset.shape, input_range_left, dtype=x_dtype)
    else:
        if x_dtype.name in ('float16', 'bfloat16'):
            antiquant_offset = np.random.randint(input_range_left, input_range_right,
                                                 size=antiquant_offset.shape).astype(x_dtype)
    return antiquant_offset
    
def weight_quant_bmmv2_uint64_quant_scale_generate(scale_shape, testcase_name):
    fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    fp32_quant_offset = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    np.save(testcase_name + "_scale.npy", fp32_scale)
    np.save(testcase_name + "_quant_offset.npy", fp32_quant_offset)
    quant_pre = []
    for idx in range(fp32_scale.flatten().size):
        quant_pre.append(get_quant_pre(fp32_scale.flatten()[idx], fp32_quant_offset.flatten()[idx]))
    quant_pre = np.array(quant_pre, dtype=np.uint64).reshape(scale_shape)
    return quant_pre

def get_quant_pre(scale, offset):
    # convert float32 to uint32
    scale_binary = struct.pack('f', scale)
    scale_int = int.from_bytes(scale_binary, byteorder='little')
    # round to nearest, tie to even
    offset_round = round(offset)
    offset_round_clip = min(max(-256, offset_round), 255)
    offset_binary = struct.pack('i', offset_round_clip)
    offset_int = int.from_bytes(offset_binary, byteorder='little') & 0x1FF  # get complement of int9

    quant_pre_u64 = (1 << 46) + (offset_int << 37) + scale_int

    return quant_pre_u64


# adapt fixpipe perf, process antiquant_scale uint64 scene
def weight_quant_bmmv2_uint64_antiquant_scale_generate(antiquant_scale, input_data_ranges, testcase_name):
    scale_shape = antiquant_scale.shape

    if (len(input_data_ranges) >= 3):
        input_range_left, input_range_right = input_data_ranges[2] # 获取antiquant_scale的range范围
    else:
        input_range_left, input_range_right = input_data_ranges[1] # 取第一个range范围

    input_range_left, input_range_right = process_quant_inf_nan(input_range_left, input_range_right, -65504, 65504) # inf nan情况下上下限处理

    fp32_scale = np.random.uniform(low=input_range_left, high=input_range_right, size=scale_shape).astype(np.float16).astype(np.float32)
    np.save(testcase_name + "_antiscale.npy", fp32_scale)
    quant_pre = []
    for idx in range(fp32_scale.flatten().size):
        quant_pre.append(get_quant_pre(fp32_scale.flatten()[idx], 0))
    quant_pre = np.array(quant_pre, dtype=np.uint64).reshape(scale_shape)
    return quant_pre

def process_quant_inf_nan(input_range_left, input_range_right, low, high):
    if ("-inf" in str(input_range_left)):
        input_range_left = low
    elif ("inf" in str(input_range_left)):
        input_range_left = high
    elif ("nan" in str(input_range_left)):
        input_range_left = 0
    else:
        input_range_left = input_range_left

    if ("-inf" in str(input_range_right)):
        input_range_right = low
    elif ("inf" in str(input_range_right)):
        input_range_right = high
    elif ("nan" in str(input_range_left)):
        input_range_right = 0
    else:
        input_range_right = input_range_right

    return input_range_left, input_range_right