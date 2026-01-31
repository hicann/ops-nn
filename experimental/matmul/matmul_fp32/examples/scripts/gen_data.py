#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import os


def gen_golden_data():
    M = 4096
    N = 128
    K = 128

    input_a = np.random.uniform(low=0, high=1, size=(M, K)).astype(np.float32)
    input_b = np.random.uniform(low=0, high=1, size=(K, N)).astype(np.float32)
    input_bias = np.random.uniform(low=0, high=1, size=N).astype(np.float32)
    golden = (np.matmul(input_a, input_b) + input_bias).astype(np.float32)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")
    input_a.tofile("./input/input_a.bin")
    input_b.tofile("./input/input_b.bin")
    input_bias.tofile("./input/input_bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
