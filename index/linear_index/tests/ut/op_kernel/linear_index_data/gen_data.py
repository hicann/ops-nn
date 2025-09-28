#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import torch
import tensorflow as tf

def gen_golden_data_simple(dtype):
    var = torch.randn(65535,4096).numpy()
    indices = torch.randint(0,65535,(63806,)).numpy()

    if dtype == "int32":
        indices = indices.astype(np.int32)

    if dtype == "int64":
        indices = indices.astype(np.int64)

    var.tofile("./var.bin")
    indices.tofile("./indices.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
