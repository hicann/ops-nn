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
import numpy as np
import torch


def swish(x):
    return x*torch.sigmoid(x)

def swish_grad(x):
    return torch.sigmoid(x) + x*(1-torch.sigmoid(x))*torch.sigmoid(x)

def gen_golden_data_simple(n, l, dtype):

    data_y_grad = np.random.uniform(-2, 2, [int(n), int(l / 2)]).astype(dtype)
    data_x = np.random.uniform(-2, 2, [int(n), int(l)]).astype(dtype)

    tensor_y_grad = torch.from_numpy(data_y_grad)
    tensor_x = torch.from_numpy(data_x)

    a, b = tensor_x.chunk(2, dim=-1)

    a = a.to(torch.float32)
    b = b.to(torch.float32)
    tensor_y_grad = tensor_y_grad.to(torch.float32)

    y1 = b * tensor_y_grad * swish_grad(a)
    y2 = tensor_y_grad * swish(a)

    y = torch.cat((y1,y2), dim = -1)
    y.to(torch.float16)

    res = y.numpy()

    data_x.tofile("./input_x.bin")
    res.tofile("./output_y_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3])