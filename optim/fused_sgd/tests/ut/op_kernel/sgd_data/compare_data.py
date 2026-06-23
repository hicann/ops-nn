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


def compare_data(d_type):
    if d_type == "float16":
        np_dtype = np.float16
        precision = 1 / 1000
    elif d_type == "float32":
        np_dtype = np.float32
        precision = 1 / 10000
    else:
        np_dtype = np.float32
        precision = 1 / 1000

    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print(f"PASSED! {os.path.basename(gold)} vs {os.path.basename(out)}")
        else:
            print(f"FAILED! {os.path.basename(gold)} vs {os.path.basename(out)}")
            for idx in diff_idx[:5]:
                print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False

    if not data_same:
        exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Param num must be 2.")
        exit(1)
    compare_data(sys.argv[1])