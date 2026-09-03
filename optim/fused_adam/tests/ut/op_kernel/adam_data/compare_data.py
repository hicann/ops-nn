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


def read_bf16(filename):
    """Read bfloat16 binary file as float32 array."""
    u16 = np.fromfile(filename, np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def compare_data(d_type):
    if d_type == "float16":
        np_dtype = np.float16
        precision = 2 / 1000
    elif d_type == "float32":
        np_dtype = np.float32
        precision = 1 / 10000
    elif d_type == "bfloat16":
        np_dtype = None  # 特殊处理
        precision = 1 / 100  # BF16 尾数仅7位，1 ULP在数值5附近约0.03125
    else:
        np_dtype = np.float32
        precision = 1 / 1000

    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))

    data_same = True
    for gold in golden_file_lists:
        # 通过文件名匹配: golden_t_xxx.bin -> output_t_xxx.bin
        basename = os.path.basename(gold)
        tensor_name = basename.replace("_golden_", "_output_")
        out = os.path.join(curr_dir, tensor_name)
        if not os.path.exists(out):
            print(f"SKIP {basename}: no matching output file")
            continue
        if d_type == "bfloat16":
            tmp_out = read_bf16(out)
            tmp_gold = read_bf16(gold)
        else:
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)
        if d_type == "bfloat16":
            # BF16 尾数仅7位，小数值时 rtol 失效，需 atol 兜底
            diff_res = np.isclose(
                tmp_out, tmp_gold, rtol=1 / 100, atol=1 / 1000, equal_nan=True
            )
        else:
            diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print(f"PASSED! {os.path.basename(gold)} vs {os.path.basename(out)}")
        else:
            print(f"FAILED! {os.path.basename(gold)} vs {os.path.basename(out)}")
            for idx in diff_idx[:5]:
                print(
                    f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}"
                )
            data_same = False

    if not data_same:
        exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Param num must be 2.")
        exit(1)
    compare_data(sys.argv[1])
