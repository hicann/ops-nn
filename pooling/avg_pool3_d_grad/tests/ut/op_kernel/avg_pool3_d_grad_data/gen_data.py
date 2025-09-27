#!/usr/bin/env python3
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
import os
import numpy as np
import torch

def gen_data_and_golden(d, h, w, nc, dtype, k):
    grads_data = np.random.uniform(-2, 2, [int(d), int(h), int(w), int(nc)]).astype(dtype)
    grads_tensor = torch.from_numpy(grads_data.astype("float32"))
    grads_tensor = grads_tensor.permute(3, 0, 1, 2)
    grads_tensor = grads_tensor.reshape(1, grads_tensor.shape[0], grads_tensor.shape[1], grads_tensor.shape[2], grads_tensor.shape[3])

    input_d = int(d) * int(k)
    input_h = int(h) * int(k)
    input_w = int(w) * int(k)
    input_tensor = torch.ones(1, int(nc), input_d, input_h, input_w, requires_grad=True).to(torch.float32)

    orig_input_shape = torch.tensor([1, int(nc), input_d, input_h, input_w], dtype=torch.int32)

    m = torch.nn.AvgPool3d(int(k))
    output = m(input_tensor)
    output.backward(grads_tensor)

    golden_tensor = input_tensor.grad #NCDHW
    golden_tensor = golden_tensor.reshape(golden_tensor.shape[1], golden_tensor.shape[2], golden_tensor.shape[3],
                                          golden_tensor.shape[4])
    golden_tensor = golden_tensor.permute(1, 2, 3, 0)
    golden_data = golden_tensor.numpy()
    orig_input_shape_data = orig_input_shape.numpy()

    grads_data.tofile("./grads.bin")
    golden_data.tofile("./golden.bin")
    orig_input_shape_data.tofile("./orig-input-shape.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    exit(0)