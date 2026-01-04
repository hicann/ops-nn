#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 联通（广东）产业互联网有限公司.
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
        rtol = 1.e-3
        atol = 1.e-3
    elif d_type == "float32":
        np_dtype = np.float32
        rtol = 1.e-4
        atol = 1.e-4
    elif d_type == "bfloat16":
        # numpy does not support bfloat16, using float32 as placeholder if converted beforehand
        np_dtype = np.float32
        rtol = 1.e-1
        atol = 1.e-1
    else:
        np_dtype = np.float32
        rtol = 1.e-4
        atol = 1.e-4
    
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        try:
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)
        except Exception as e:
            print(f"Read file failed: {e}")
            continue

        if tmp_out.shape != tmp_gold.shape:
            print(f"Shape mismatch in {os.path.basename(out)}: output {tmp_out.shape} vs golden {tmp_gold.shape}")
            data_same = False
            continue

        diff_res = np.isclose(tmp_out, tmp_gold, rtol=rtol, atol=atol, equal_nan=True)
        
        print(f"=== Preview first 5 elements for {os.path.basename(out)} ===")
        for idx in range(min(5, tmp_out.size)):
            print(f"index: {idx}, output: {tmp_out[idx]:.30f}, golden: {tmp_gold[idx]:.30f}")

        if not np.all(diff_res):
            data_same = False
            diff_idx = np.where(diff_res == False)[0]
            print(f"Comparison failed for {os.path.basename(out)}")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
        else:
            print(f"Comparison passed for {os.path.basename(out)}")

    if data_same:
        print("PASSED!")
    else:
        print("FAILED!")
    return data_same

def get_file_lists():
    golden_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*golden*.bin")))
    output_file_lists = sorted(glob.glob(os.path.join(curr_dir, "*output*.bin")))
    return golden_file_lists, output_file_lists

def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists()
    if len(golden_file_lists) != len(output_file_lists) or len(golden_file_lists) == 0:
        print("Error: File count mismatch or no files found.")
        print(f"Golden files: {len(golden_file_lists)}")
        print(f"Output files: {len(output_file_lists)}")
        return
    compare_data(golden_file_lists, output_file_lists, d_type)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 verify_result.py <dtype>")
    else:
        process(sys.argv[1])