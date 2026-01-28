#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import torch
import tensorflow as tf
bfloat16 = tf.bfloat16.as_numpy_dtype

case_list = {
    "case1" : {"x_shape":[5, 2, 32, 128], "weight_shape":[2]},
    "case2" : {"x_shape":[170, 2, 2, 125], "weight_shape":[2]},
    "case3" : {"x_shape":[8, 2, 8, 16, 16], "weight_shape":[2]},
    "case4" : {"x_shape":[32, 2, 13, 16], "weight_shape":[8]},
    "case5" : {"x_shape":[32, 2, 16, 16], "weight_shape":[4]},
    "case6" : {"x_shape":[32, 2, 16, 16], "weight_shape":[8]},
    "case7" : {"x_shape":[1, 2, 448, 448], "weight_shape":[2]},
    "case8" : {"x_shape":[10, 2, 192, 108], "weight_shape":[2]},
    "case9" : {"x_shape":[20, 2, 28, 28, 16], "weight_shape":[2]},
    "infer_channel_last_fp32": {"x_shape":[10, 2, 3, 200], "weight_shape":[200], "is_training": False, "epsilon": 0, "format": "NHWC"},
    "infer_fp32": {"x_shape":[1, 2, 1, 1], "weight_shape":[2], "is_training": False, "epsilon": 0, "format": "NCHW"},
}

def batch_norm_grad_v3(dy, x, weight, running_mean, running_var, save_mean, save_rstd, training=True, epsilon=1e-5, format="NCHW"):
    dtype = dy.dtype
    if dtype != np.float32:
        dy = dy.astype(np.float32)
        x = x.astype(np.float32)
        weight = weight.astype(np.float32)
        running_mean = running_mean.astype(np.float32)
        running_var = running_var.astype(np.float32)
    tensor_dy = torch.from_numpy(dy).to(torch.float32)
    tensor_x = torch.from_numpy(x).to(torch.float32)
    tensor_weight = torch.from_numpy(weight).to(torch.float32)
    tensor_running_mean = torch.from_numpy(running_mean).to(torch.float32)
    tensor_running_var = torch.from_numpy(running_var).to(torch.float32)
    tensor_save_mean = torch.from_numpy(save_mean).to(torch.float32)
    tensor_save_rstd = torch.from_numpy(save_rstd).to(torch.float32)

    if format == "NHWC":
        tensor_dy = tensor_dy.permute(0, 3, 1, 2)  #NHWC -> NCHW
        tensor_x = tensor_x.permute(0, 3, 1, 2)
    elif format == "NDHWC":
        tensor_dy = tensor_dy.permute(0, 4, 1, 2, 3)  #NDHWC -> NCDHW
        tensor_x = tensor_x.permute(0, 4, 1, 2, 3)

    res = torch.ops.aten.native_batch_norm_backward(grad_out=tensor_dy, input=tensor_x,
        weight=tensor_weight, running_mean=tensor_running_mean, running_var=tensor_running_var,
        save_mean=tensor_save_mean, save_invstd=tensor_save_rstd, train=training, eps=epsilon,
        output_mask=[True, True, True])

    dx = res[0].numpy().astype(dtype)
    dgamma = res[1].numpy().astype(dtype)
    dbeta = res[2].numpy().astype(dtype)

    if format == "NHWC":
        dx = np.transpose(dx, (0, 2, 3, 1))  #NCHW -> NHWC
    elif format == "NDHWC":
        dx = np.transpose(dx, (0, 2, 3, 4, 1))  #NCDHW -> NDHWC

    return dx, dgamma, dbeta


def gen_golden_data_simple(param, dtype):
    dy = np.random.uniform(-2, 2, param["x_shape"]).astype(dtype).reshape(param["x_shape"])
    x = np.random.uniform(-2, 2, param["x_shape"]).astype(dtype).reshape(param["x_shape"])
    weight = np.random.uniform(-2, 2, param["weight_shape"]).astype(dtype).reshape(param["weight_shape"])
    running_mean = np.random.uniform(-2, 2, param["weight_shape"]).astype(np.float32).reshape(param["weight_shape"])
    running_var = np.random.uniform(1, 1, param["weight_shape"]).astype(np.float32).reshape(param["weight_shape"])
    save_mean = np.random.uniform(-2, 2, param["weight_shape"]).astype(np.float32).reshape(param["weight_shape"])
    save_rstd = np.random.uniform(-2, 2, param["weight_shape"]).astype(np.float32).reshape(param["weight_shape"])

    is_training = param.get('is_training', True)
    epsilon = param.get('epsilon', 1e-5)
    format = param.get('format', "NCHW")

    dy.tofile("input_dy.bin")
    x.tofile("input_x.bin")
    weight.tofile("input_weight.bin")
    running_mean.tofile("input_running_mean.bin")
    running_var.tofile("input_running_var.bin")
    save_mean.tofile("input_save_mean.bin")
    save_rstd.tofile("input_save_rstd.bin")

    dx_golden, dgamma_golden, dbeta_golden = batch_norm_grad_v3(dy, x, weight, running_mean, running_var, save_mean, save_rstd, is_training, epsilon, format)
    dx_golden.tofile("dx_golden.bin")
    dgamma_golden.tofile("dgamma_golden.bin")
    dbeta_golden.tofile("dbeta_golden.bin")


DTYPE_MAP = {
    "float16":np.float16,
    "float":np.float32,
    "float32": np.float32,
    "bfloat16": bfloat16
}

if __name__ == "__main__":
    case_name = sys.argv[1]
    dtype = sys.argv[2]
    print("[case_name]: ", case_name)
    print("[dtype]: ", dtype)
    param = case_list.get(case_name)
    dtype = DTYPE_MAP.get(dtype)
    print("param:", param)
    gen_golden_data_simple(param, dtype)