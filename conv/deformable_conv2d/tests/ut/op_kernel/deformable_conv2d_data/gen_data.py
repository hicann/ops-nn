#!/usr/bin/python3
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
import os
import torch
import torch_npu
import numpy as np
import tensorflow as tf

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return shape_list

def gen_data_and_golden(x_shape_str, weight_shape_str, offset_shape_str, bias_shape_str, type_str="float32"):
    np_dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }
    np_dtype = np_dtype_dict[type_str]
    x_shape = parse_str_to_shape_list(x_shape_str)
    weight_shape = parse_str_to_shape_list(weight_shape_str)
    offset_shape = parse_str_to_shape_list(offset_shape_str)
    bias_shape = parse_str_to_shape_list(bias_shape_str)

    x = np.ones(x_shape).astype(np_dtype)
    weight = np.ones(weight_shape).astype(np_dtype)
    offset = np.ones(offset_shape).astype(np_dtype)
    bias = np.ones(bias_shape).astype(np_dtype)

    x.tofile(f"{type_str}_x_deformable_conv2d.bin")
    weight.tofile(f"{type_str}_weight_deformable_conv2d.bin")
    offset.tofile(f"{type_str}_offset_deformable_conv2d.bin")
    bias.tofile(f"{type_str}_bias_deformable_conv2d.bin")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Param num must be 6, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
