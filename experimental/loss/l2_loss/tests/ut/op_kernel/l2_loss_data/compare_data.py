#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(golden_file_lists, output_file_lists, d_type):
    if d_type == "float16":
        np_dtype = np.float16
    elif d_type == "float32":
        np_dtype = np.float32
    else:
        raise ValueError("d_type must be float16 or float32")

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, rtol=1e-3, atol=1e-5, equal_nan=True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                abs_diff = abs(float(tmp_out[idx]) - float(tmp_gold[idx]))
                rel_diff = abs_diff / (abs(float(tmp_gold[idx])) + 1e-12)
                print(f"index: [{idx}]  real output value: [{tmp_out[idx]}] expected value: [{tmp_gold[idx]}]"
                      f"  abs_diff={abs_diff:.6g}  rel_diff={rel_diff:.3e}")
            data_same = False
    return data_same

def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result

if __name__ == '__main__':
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
