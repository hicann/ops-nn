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
import torch
import numpy as np
from bfloat16 import bfloat16


def gen_golden_data(x_shape, x_dtype, x_format, beta_shape, epsilon, mean_dtype):
    x_shape = eval(x_shape)
    beta_shape = eval(beta_shape)

    if x_dtype == "bfloat16":
        x = np.random.uniform(2, 2, x_shape).astype(bfloat16)
        if mean_dtype == "bfloat16":
            mean = np.random.uniform(1, 1, beta_shape).astype(bfloat16)
            var = np.random.uniform(1, 1, beta_shape).astype(bfloat16);
        else:
            mean = np.random.uniform(1, 1, beta_shape).astype(mean_dtype)
            var = np.random.uniform(1, 1, beta_shape).astype(mean_dtype);
        beta = np.random.uniform(2, 2, beta_shape).astype(bfloat16);
        gamma = np.random.uniform(2, 2, beta_shape).astype(bfloat16);
    else:
        x = np.random.uniform(2, 2, x_shape).astype(x_dtype)
        if mean_dtype == "bfloat16":
            mean = np.random.uniform(1, 1, beta_shape).astype(bfloat16)
            var = np.random.uniform(1, 1, beta_shape).astype(bfloat16);
        else:
            mean = np.random.uniform(1, 1, beta_shape).astype(mean_dtype)
            var = np.random.uniform(1, 1, beta_shape).astype(mean_dtype);
        beta = np.random.uniform(2, 2, beta_shape).astype(x_dtype);
        gamma = np.random.uniform(2, 2, beta_shape).astype(x_dtype);

    x.tofile("./x.bin")
    mean.tofile("./mean.bin")
    var.tofile("./var.bin")
    beta.tofile("./beta.bin")
    gamma.tofile("./gamma.bin")

    promote_dtype = "float32" if x_dtype in ("bfloat16", "float16") else "float64"
    x = x.astype(promote_dtype)
    mean = mean.astype(promote_dtype)
    var = var.astype(promote_dtype)
    beta = beta.astype(promote_dtype)
    gamma = gamma.astype(promote_dtype)

    tensor_x = torch.from_numpy(x)
    tensor_mean = torch.from_numpy(mean)
    tensor_var = torch.from_numpy(var)
    tensor_beta = torch.from_numpy(beta)
    tensor_gamma = torch.from_numpy(gamma)

    if x_format == "NHWC":
        tensor_x = tensor_x.permute(0,3,1,2)
    if x_format == "NDHWC":
        tensor_x = tensor_x.permute(0,4,1,2,3)

    res = torch.ops.aten.native_batch_norm(input=tensor_x, weight=tensor_gamma, bias=tensor_beta,
                                           running_mean=tensor_mean, running_var=tensor_var,
                                           training=False, momentum=0.1, eps=float(epsilon))

    y = res[0]
    if x_format == "NHWC" :
        y = y.permute(0, 2,3,1)
    if x_format == "NDHWC":
        y = y.permute(0,2,3,4,1)
    y =y.numpy().astype(x_dtype)

    y.tofile("./golden_y.bin")

    running_mean = tensor_mean.numpy()
    running_mean = running_mean.astype(mean_dtype)
    running_var = tensor_var.numpy()
    running_var = running_var.astype(mean_dtype)
    save_mean = res[1].numpy()

    running_mean.tofile("./golden_running_mean.bin")
    running_var.tofile("./golden_running_var.bin")
    save_mean.tofile("./golden_save_mean.bin")
if __name__ == "__main__":
    gen_golden_data(*sys.argv[1:])
