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

def gen_golden_data_simple(n, l, dtype):
    # np.random.seed(1)
    data_dy = np.random.uniform(-2, 2, [int(n), int(l)]).astype(dtype)
    data_x1 = np.random.uniform(-2, 2, [int(n), int(l)]).astype(dtype)
    data_x2 = np.random.uniform(-2, 2, [int(n), int(l)]).astype(dtype)
    data_rstd = np.random.uniform(-2, 2, [int(n)]).astype(dtype)
    data_mean = np.random.uniform(-2, 2, [int(n)]).astype(dtype)
    data_gamma = np.random.uniform(-2, 2, [int(l)]).astype(dtype)
    data_dsum = np.random.uniform(-2, 2, [int(l)]).astype(dtype)

    dx = data_x1
    dgamma = data_gamma
    dbeta = data_gamma
    data_dy.tofile("./input_dy.bin")
    data_x1.tofile("./input_x1.bin")
    data_x2.tofile("./input_x2.bin")
    data_rstd.tofile("./input_rstd.bin")
    data_mean.tofile("./input_mean.bin")
    data_gamma.tofile("./input_gamma.bin")
    data_dsum.tofile("./input_dsum.bin")

    dx.tofile("./output_dx_golden.bin")
    dgamma.tofile("./output_dgamma_golden.bin")
    dbeta.tofile("./output_dbeta_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3])
