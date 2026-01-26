#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import tensorflow as tf


def compare_data(tensor_count, tiling_key="1"):
    if tiling_key == "2":
        d_type = np.float16
    elif tiling_key == "3":
        d_type = tf.bfloat16.as_numpy_dtype
    else:
        d_type = np.float32
    data_same = True
    print("===============compare data start==============")
    for i in range(int(tensor_count)):
        tmp_output = np.fromfile(f"output_t{i}.bin", d_type)
        tmp_golden = np.fromfile(f"golden_t{i}.bin", d_type)
        if d_type == np.float32:
            precision_value = 1/10000
        else:
            precision_value = 1/1000
        print(f"===============tensor[{i}]==============")
        for j in range(len(tmp_golden)):
            if abs(tmp_golden[j]-tmp_output[j]) > precision_value:
                print(f"index:{j}, golden:{tmp_golden[j]}, output:{tmp_output[j]}")
                data_same = False
    print("===============compare data finish==============")
    return data_same


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)

    if compare_data(sys.argv[1], sys.argv[2]):
        exit(0)
    else:
        exit(2)
