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
from bfloat16 import bfloat16

def compare_data(y_dtype):
    y = np.fromfile("./y.bin", y_dtype)
    golden_y = np.fromfile("./golden_y.bin", y_dtype)

    check_y = np.allclose(y, golden_y, 0.005, 0.005)
    if not check_y:
        raise RuntimeError("y compare failed")

if __name__ == "__main__":
    compare_data(*sys.argv[1:])