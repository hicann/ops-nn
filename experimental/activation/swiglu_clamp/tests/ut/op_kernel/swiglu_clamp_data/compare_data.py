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

# SwigluClamp kernel UT comparator.
# Compares the kernel output bin against the golden bin produced by gen_data.py.
# Uses rtol (sigmoid/silu implementation may differ slightly between AscendC and
# numpy; a small tolerance catches real bugs — wrong silu/clamp order, swapped
# gate/up, missing clamp — without flagging float noise).
#
# usage: compare_data.py <float16|float32>

import sys
import os
import glob
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare(dtype):
    if dtype == "float16":
        np_dtype = np.float16
        rtol = 1e-3
        atol = 1e-3
    elif dtype == "float32":
        np_dtype = np.float32
        rtol = 1e-4
        atol = 1e-4
    else:
        raise ValueError("dtype must be float16 or float32")

    golden_files = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_files = sorted(glob.glob(curr_dir + "/*output*.bin"))

    data_same = True
    for gold_f, out_f in zip(golden_files, output_files):
        gold = np.fromfile(gold_f, np_dtype)
        out = np.fromfile(out_f, np_dtype)
        if out.shape != gold.shape:
            print(
                "FAILED! shape mismatch: output {} golden {}".format(
                    out.shape, gold.shape
                )
            )
            data_same = False
            continue
        if not np.allclose(out, gold, rtol=rtol, atol=atol, equal_nan=False):
            diff = np.abs(out.astype(np.float32) - gold.astype(np.float32))
            idx = int(np.argmax(diff))
            print(
                "FAILED! idx {} output {} golden {} (max diff {})".format(
                    idx, out[idx], gold[idx], diff[idx]
                )
            )
            data_same = False
        else:
            print("PASSED!")
    return data_same


if __name__ == "__main__":
    ret = compare(sys.argv[1])
    print("compare result:", ret)
    exit(0 if ret else 1)
