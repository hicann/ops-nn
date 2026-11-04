#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def _read_as_fp32(path, d_type):
    """bfloat16_t 落盘为 2 字节载体, 按 uint16 读入再按位左移还原 float32;
    其余 dtype 按原生宽度读取。"""
    if d_type == "bfloat16_t":
        raw = np.fromfile(path, np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if d_type == "float16":
        return np.fromfile(path, np.float16).astype(np.float32)
    return np.fromfile(path, np.float32)


def compare_data(golden_file_lists, output_file_lists, d_type):
    if d_type == "float16":
        precision = 1 / 1000
    elif d_type == "float32":
        precision = 1 / 10000
    else:
        # bfloat16 尾数 8 bit, 相对精度约 2^-8; sigmoid 输出恒在 [0,1], 1/10 过宽会漏报
        precision = 1 / 100

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = _read_as_fp32(out, d_type)
        tmp_gold = _read_as_fp32(gold, d_type)
        if tmp_out.shape != tmp_gold.shape:
            print("FAILED!")
            print(
                f"element count mismatch: output {tmp_out.shape}, golden {tmp_gold.shape}"
            )
            data_same = False
            continue
        diff_res = np.isclose(
            tmp_out, tmp_gold, rtol=precision, atol=0.0, equal_nan=True
        )
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


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    return 0 if compare_data(golden_file_lists, output_file_lists, d_type) else 1


if __name__ == "__main__":
    sys.exit(process(sys.argv[1]))
