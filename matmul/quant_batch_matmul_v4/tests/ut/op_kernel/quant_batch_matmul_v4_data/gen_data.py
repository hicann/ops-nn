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


def quant_batch_matmul_v4(m, k, n, group):
    np.random.random([m, k]).astype(np.int8).tofile("x.bin")
    np.random.uniform(-1, 1, [n, k // 2]).astype(np.int8).tofile("weight.bin")
    np.random.random([n, k // group]).astype(np.float16).tofile("antiquant_scale.bin")
    np.random.random([1, n]).astype(np.float32).tofile("quant_scale.bin")

def quant_batch_matmul_v4_mx(m, k, n, group):
    np.random.random([m, k]).astype(np.int8).tofile("x1.bin")
    np.random.uniform(-1, 1, [n, k // 2]).astype(np.int8).tofile("x2.bin")
    np.random.random([1, n]).astype(np.float16).tofile("bias.bin")
    np.random.random([m, k // group]).astype(np.float16).tofile("x1_scale.bin")
    np.random.random([n, k // group]).astype(np.float16).tofile("x2_scale.bin")

if __name__ == '__main__':
    m, k, n, group, quant_type = [int(p) for p in sys.argv[1:]]
    if quant_type == 3:
        quant_batch_matmul_v4(m, k, n, group)
    elif quant_type == 4:
        quant_batch_matmul_v4_mx(m, k, n, group)
