#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import os
import glob
import numpy as np

try:
    import ml_dtypes
    BFLOAT16 = ml_dtypes.bfloat16
except ImportError:
    BFLOAT16 = None

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
    }
    if BFLOAT16 is not None:
        dtype_map["bfloat16_t"] = BFLOAT16
    np_dtype = dtype_map.get(d_type, np.float32)

    if d_type == "float32":
        precision = 1e-4
    else:
        precision = 1e-2

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print("COMPARE DATA PASSED!")
        else:
            print("COMPARE DATA FAILED!")
            print(f"  first fail idx: {diff_idx[0]}, output: {tmp_out[diff_idx[0]]}, "
                  f"golden: {tmp_gold[diff_idx[0]]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + f"/{dtype}_golden.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + f"/{dtype}_npuout.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    if not golden_file_lists or not output_file_lists:
        print(f"COMPARE DATA SKIPPED! Missing files for dtype={d_type}")
        sys.exit(1)
    data_same = compare_data(golden_file_lists, output_file_lists, d_type)
    if not data_same:
        sys.exit(1)


if __name__ == '__main__':
    process(sys.argv[1])
