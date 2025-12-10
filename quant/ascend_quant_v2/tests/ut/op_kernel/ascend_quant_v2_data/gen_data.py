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
import os
import numpy as np


def gen_data_and_golden(dtype_x):
    data_x = np.random.uniform(-2, 2, [10240, 1023]).astype(dtype_x)
    data_scale = np.random.uniform(-3, 2, [1023]).astype(dtype_x)
    data_offset = np.random.uniform(-3, 4, [1023]).astype(dtype_x)
    data_x.tofile("./x.bin")
    data_scale.tofile("./scale.bin")
    data_offset.tofile("./offset.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1])
    exit(0)