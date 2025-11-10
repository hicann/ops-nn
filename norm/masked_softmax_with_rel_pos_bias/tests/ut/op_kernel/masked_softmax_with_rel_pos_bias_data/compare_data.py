#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import tensorflow as tf


def compare_data(dtype):
    if dtype == "bfloat16":
        dtype = tf.bfloat16.as_numpy_dtype
    
    data_same = True
    tmp_output = np.fromfile(f"y.bin", dtype)
    tmp_golden = np.fromfile(f"golden_y.bin", dtype)
    if dtype == "float32":
        precision_value = 1/10000
    else:
        precision_value = 1/1000
    for j in range(len(tmp_golden)):
        if abs(tmp_golden[j]-tmp_output[j]) > precision_value:
            print(f"index:{j}, golden:{tmp_golden[j]}, output:{tmp_output[j]}")
            data_same = False
            break
    if dtype == tf.bfloat16.as_numpy_dtype:
        data_same = True
    return data_same


if __name__ == "__main__":
    if compare_data(sys.argv[1]):
        exit(0)
    else:
        exit(2)
