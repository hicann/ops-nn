#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": None,
    }
    np_dtype = dtype_map.get(d_type, np.float32)

    if d_type == "bfloat16":
        import ml_dtypes

        np_dtype = ml_dtypes.bfloat16
        precision = 1 / 100
    elif d_type == "float16":
        precision = 1 / 100
    elif d_type == "int32":
        precision = 0
    elif d_type == "float32":
        precision = 1 / 10000
    else:
        precision = 1 / 100

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        if d_type == "int32":
            diff_res = tmp_out == tmp_gold
            diff_idx = np.where(~diff_res)[0]
        else:
            diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
            diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


if __name__ == "__main__":
    golden_lists, output_lists = get_file_lists(sys.argv[1])
    result = compare_data(golden_lists, output_lists, sys.argv[1])
    sys.exit(0 if result else 1)
