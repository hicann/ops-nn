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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def gen_input_data(shape, target, weight, dtype, input_range):
    x = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    y = np.random.uniform(input_range[0], shape[1], target).astype(np.int64)
    z = np.random.uniform(input_range[0], input_range[1], weight).astype(np.float32)
    return x, y, z

def gen_golden_data_simple(shape, target, weight, dtype, input_range, reduction):
    predict, target, weight = gen_input_data(shape, target, weight, dtype, input_range)
    import torch
    import torch.nn as nn
    predict = predict.astype(np.float32)
    x = torch.from_numpy(predict)
    target = torch.from_numpy(target)
    weight = torch.from_numpy(weight)
    crossentropyloss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    loss = crossentropyloss(x, target)
    predict.tofile("./predict.bin")
    target.numpy().tofile("./target.bin")
    weight.numpy().tofile("./weight.bin")
    loss.numpy().tofile("./golden.bin")

case_list = {
    "test_case_0" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"sum",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_1" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"mean",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_2" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"none",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_3" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"sum",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_4" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"mean",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_5" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"none",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_6" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"sum",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_7" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"mean",
                     "target_shape":[4], "weight_shape": [64]},
    "test_case_8" : {"shape":[4, 64], "input_range":[0, 0.5], "reduction":"none",
                     "target_shape":[4], "weight_shape": [64]},
    }

if __name__ == "__main__":
    case_name = sys.argv[1]
    dtype = sys.argv[2]
    param = case_list.get(case_name)
    d_type_dict = {
        "float32": np.float32,
        "float": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
    }
    dtype = d_type_dict.get(dtype)
    gen_golden_data_simple(param["shape"], param["target_shape"], param["weight_shape"], dtype, param["input_range"], param["reduction"])