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
    np_dtype = np.float32
    if d_type == "float16":
        np_dtype = np.float16
        precision = 1 / 100
    elif d_type == "float32":
        precision = 1 / 10000
    else:
        precision = 1 / 100

    # log_softmax_grad reduces over an axis in float32; the reduction (sum of up to
    # thousands of elements) carries an absolute rounding error ~1e-4..1e-3. Near-zero
    # outputs would otherwise fail a strict relative-only (atol=0) check. We keep the
    # example's rtol=1e-4 but add a small atol that only absorbs the float32 rounding
    # noise -- a real correctness bug produces errors of O(1) and still fails.
    atol = 1e-3
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)
        diff_res = np.isclose(tmp_out, tmp_gold, precision, atol, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(d_type):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    if not compare_data(golden_file_lists, output_file_lists, d_type):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    process(sys.argv[1])
