#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import functools


def cal_relative_diff(x, y):
    if abs(float(x) - float(y)) < 0.001:
        result = abs(float(x) - float(y))
    else:
        result = abs(float(x) - float(y)) / (float(max(abs(x), abs(y))) + 10e-10)

    return result


def data_compare(data_check, data_expect, diff_thd=0.01, pct_thd=0.05):
    start = 0
    end = len(data_expect) - 1
    rdiff = list(
        map(lambda x, y: cal_relative_diff(x, y), data_check.astype(np.float32), data_expect.astype(np.float32)))
    spilt_rdiff = rdiff[start:end+1]
    spilt_count = int(end - start + 1) if end != start else 1
    lt_num = functools.reduce(lambda x, y: x + 1 if diff_thd else x, spilt_rdiff, 0)
    lt_pct = float(lt_num) / float(spilt_count) * 100.0
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (lt_pct >= pct_thd) else "Failed"
    print("\n--" * 100)
    print("DiffThd  \t PctThd   \t PctRlt   \t Result")
    print("\n--" * 100)
    print("%.3f     \t %.2f%%   \t %.6f%%   \t %s    \n" % (diff_thd, pct_thd, lt_pct, result))


def diff_data():
    cpu_file = np.fromfile("./out.bin", dtype=np.float32)
    npu_file = np.fromfile("./npu.bin", dtype=np.float32)
    data_compare(npu_file, cpu_file, 0.0001, 0.0001)


if __name__ == '__main__':
    diff_data()
