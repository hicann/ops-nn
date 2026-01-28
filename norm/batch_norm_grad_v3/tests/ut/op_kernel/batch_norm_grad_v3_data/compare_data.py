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
from bfloat16 import bfloat16

def compare_data(dx_dtype):
    if dx_dtype == "bfloat16":
        dx = np.fromfile("./cce_cpu_dx.bin", bfloat16).astype(np.float32)
        golden_dx = np.fromfile("./dx_golden.bin", bfloat16).astype(np.float32)
    elif dx_dtype == "float16":
        dx = np.fromfile("./cce_cpu_dx.bin", np.float16).astype(np.float32)
        golden_dx = np.fromfile("./dx_golden.bin", np.float16).astype(np.float32)
    else:
        dx = np.fromfile("./cce_cpu_dx.bin", np.float32)
        golden_dx = np.fromfile("./dx_golden.bin", np.float32)

    check_dx = np.allclose(dx, golden_dx, 0.005, 0.005)
    if not check_dx:
        raise RuntimeError("dx compare failed")

if __name__ == "__main__":
    compare_data(*sys.argv[1:])