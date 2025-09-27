#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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

def gen_golden_data_simple(size, dtype):
    data = np.random.uniform(-100, 100, [int(size)]).astype(np.float32)
    scale = np.random.uniform(-100, 100, [1]).astype(np.float32)
    zero_point = np.random.uniform(-100, 100, [1]).astype(np.int32)

    if dtype == "float16":
        data = data.astype(np.float16)
        scale = scale.astype(np.float16)

    data.tofile("./data.bin")
    scale.tofile("./scale.bin")
    zero_point.tofile("./zero_point.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2])