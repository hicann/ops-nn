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

def compare_data(tensor_count):
    data_same = True
    print("===============compare data start==============")
    for i in range(int(tensor_count)):
        tmp_output = np.fromfile(f"output_{i}.bin", np,int8)
        tmp_golden = np.fromfile(f"golden_{i}.bin", np,int8)
        precision_value = 1/1000
        print(f"===============tensor[{i}]==============")
        for j in range(len(tmp_golden)):
            if abs(tmp_golden[j] - tmp_output[j]) > precision_value:
                print(f"index:{j}, golden:{tmp_golden[j]}, output:{tmp_output[j]}")
                data_same = False
    print("===============compare data finish==============")
    return data_same

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Param num must be  2.")
        exit(1)

    if compare_data(sys.argv[1]):
        exit(0)
    else:
        exit(2)
