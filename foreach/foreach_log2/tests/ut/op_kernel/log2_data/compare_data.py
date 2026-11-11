#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


def load_bf16_as_fp32(path):
    # bf16 无原生 numpy dtype: 以 uint16 读入, 左移 16bit 补零扩展为 fp32 高位, 位级还原后比较
    u16 = np.fromfile(path, np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def compare_data(golden_file_lists, output_file_lists, d_type):
    np_dtype = np.float32
    is_bf16 = False
    if d_type == "float16":
        np_dtype = np.float16
        precision = 1 / 1000
    elif d_type == "float32":
        precision = 1 / 10000
    elif d_type == "int32":
        np_dtype = np.int32
        precision = 1 / 10000
    elif d_type in ("bfloat16", "bfloat16_t"):
        is_bf16 = True
        # bf16 尾数 8bit, 1ulp ≈ 2^-8 ≈ 0.0039, rtol 0.01 约留 2.5ulp 余量
        precision = 1 / 100
    else:
        precision = 1 / 10

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        if is_bf16:
            tmp_out = load_bf16_as_fp32(out)
            tmp_gold = load_bf16_as_fp32(gold)
        else:
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)
        # 防 isclose 广播/空数组空转: 单元素 output 会被广播成 N 对比较,
        # 两侧空文件 diff_idx 恒为空, 均会假绿 —— 先核验元素数再比较
        if tmp_out.size == 0 or tmp_gold.size == 0:
            data_same = False
            print(
                f"FAILED! file: {out if tmp_out.size == 0 else gold}, "
                f"empty file (0 element)"
            )
            continue
        if tmp_out.size != tmp_gold.size:
            data_same = False
            print(
                f"FAILED! file: {out}, element count mismatch: "
                f"golden={tmp_gold.size}, output={tmp_out.size}"
            )
            continue
        diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(~diff_res)[0]
        if diff_idx.size > 0:
            data_same = False
            print(
                f"FAILED! file: {out}, first mismatch index: {diff_idx[0]}, "
                f"expect {tmp_gold[diff_idx[0]]}, got {tmp_out[diff_idx[0]]}"
            )

    if data_same:
        print("PASSED!")
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    # 防 zip 配对空转: golden 缺失(如 gen_data 失败)或 output 缺失/部分缺失(如写文件失败)时
    # 显式失败, 而不是静默少比后按 data_same=True 通过
    if len(golden_file_lists) == 0 or len(golden_file_lists) != len(output_file_lists):
        print(
            f"FAILED! golden/output file count mismatch: "
            f"golden={len(golden_file_lists)}, output={len(output_file_lists)}"
        )
        return False
    return compare_data(golden_file_lists, output_file_lists, d_type)


if __name__ == "__main__":
    if not process(sys.argv[1]):
        sys.exit(1)
