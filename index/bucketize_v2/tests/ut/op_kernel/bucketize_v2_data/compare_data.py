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
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))

def compare_data(dtype="int64"):
    np_dtype = np.int64
    if dtype == "int32":
        np_dtype = np.int32
    
    golden_file = os.path.join(curr_dir, "res_golden.bin")
    output_file = os.path.join(curr_dir, "res_output.bin")
    
    if not os.path.exists(golden_file):
        print(f"golden file not found: {golden_file}")
        return False
    if not os.path.exists(output_file):
        print(f"output file not found: {output_file}")
        return False
    
    golden = np.fromfile(golden_file, np_dtype)
    output = np.fromfile(output_file, np_dtype)
    
    if golden.shape != output.shape:
        print(f"shape mismatch: golden {golden.shape} vs output {output.shape}")
        return False
    
    if np.array_equal(golden, output):
        print("PASSED!")
        return True
    else:
        diff_idx = np.where(golden != output)[0]
        print(f"FAILED! {len(diff_idx)} elements differ")
        for idx in diff_idx[:10]:
            print(f"index: {idx}, output: {output[idx]}, golden: {golden[idx]}")
        return False

if __name__ == '__main__':
    dtype = sys.argv[1] if len(sys.argv) > 1 else "int64"
    result = compare_data(dtype)
    sys.exit(0 if result else 1)