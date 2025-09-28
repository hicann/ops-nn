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
import tensorflow as tf
import traceback


def compare_data(tiling_key="301"):
    if tiling_key in ["101", "102", "103", "701", "702", "703"]:
        d_type = tf.bfloat16.as_numpy_dtype
        precision_value = 4 / 1000
    elif tiling_key in ["201", "202", "203", "801", "802", "803"]:
        d_type = np.float16
        precision_value = 1 / 1000
    else:
        d_type = np.float32
        precision_value = 1 / 10000
    data_same = True
    print("===============compare data start==============")
    output_dx = np.fromfile(f"output_dx.bin", d_type)
    output_golden = np.fromfile(f"output_golden.bin", d_type)

    diff_count = 0
    for j in range(len(output_golden)):
        if abs(output_golden[j] - output_dx[j]) > precision_value and diff_count < 20:
            print(f"index:{j}, golden:{output_golden[j]}, output:{output_dx[j]}")
            data_same = False
            diff_count += 1
    print("===============compare data finish==============")
    return data_same


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Param num must be 2.")
        exit(1)
    try:
        if not compare_data(sys.argv[1]):
            exit(2)
    except Exception:
        traceback.print_exc()
        exit(3)
    exit(0)
