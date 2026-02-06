#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np

CASE_LIST = {
    "test_case_nhwc_bigc_800001": {
        "n": 17, "h": 14, "w": 8, "c": 11, "hk": 5, "wk": 6,
        "sh": 3, "sw": 6, "pt": 0, "pl": 9, "ho": 4, "wo": 1, "m": "nhwc"
    },
    "test_case_nhwc_bigc_800011": {
        "n": 344, "h": 2, "w": 2, "c": 2, "hk": 2, "wk": 2,
        "sh": 8303, "sw": 2, "pt": 0, "pl": 0, "ho": 1, "wo": 1, "m": "nhwc"
    },
    "test_case_nhwc_bigc_800002": {
        "n": 17, "h": 5, "w": 7, "c": 19, "hk": 4, "wk": 2,
        "sh": 4, "sw": 1, "pt": 1, "pl": 0, "ho": 2, "wo": 7, "m": "nhwc"
    },
    "test_case_nhwc_bigc_800012": {
        "n": 2, "h": 2, "w": 1, "c": 79828, "hk": 2, "wk": 2,
        "sh": 2, "sw": 2, "pt": 0, "pl": 0, "ho": 1, "wo": 1, "m": "nhwc"
    },
    "test_case_nhwc_smallc_700001": {
        "n": 3, "h": 7, "w": 5871, "c": 3, "hk": 2, "wk": 2,
        "sh": 2, "sw": 10, "pt": 0, "pl": 0, "ho": 3, "wo": 587, "m": "nhwc"
    },
    "test_case_nhwc_smallc_700011": {
        "n": 4, "h": 2218, "w": 4, "c": 15, "hk": 2, "wk": 2,
        "sh": 2, "sw": 1068, "pt": 0, "pl": 0, "ho": 1109, "wo": 1, "m": "nhwc"
    },
    "test_case_nhwc_smallc_700002": {
        "n": 25, "h": 19, "w": 20, "c": 24, "hk": 8, "wk": 8,
        "sh": 7, "sw": 6, "pt": 1, "pl": 3, "ho": 3, "wo": 4, "m": "nhwc"
    },
    "test_case_nchw_simt_500001": {
        "n": 20, "h": 4, "w": 12, "c": 5, "hk": 3, "wk": 6,
        "sh": 3, "sw": 5, "pt": 0, "pl": 0, "ho": 1, "wo": 2, "m": "nchw"
    },
    "test_case_nchw_simt_500101": {
        "n": 3, "h": 3, "w": 3311, "c": 12, "hk": 2, "wk": 2,
        "sh": 2, "sw": 4, "pt": 0, "pl": 0, "ho": 2, "wo": 828, "m": "nchw"
    },
}


def compute_pool_core(x, p, n, c, h, w):
    ho, wo = p["ho"], p["wo"]
    mode = p["m"]
    y = np.zeros((n, ho, wo, c) if mode == "nhwc" else (n, c, ho, wo), dtype=x.dtype)
    argmax = np.zeros_like(y, dtype=np.int64)

    for i in range(n):
        for ci in range(c):
            for hi in range(ho):
                h_start = hi * p["sh"] - p["pt"]
                for wi in range(wo):
                    w_start = wi * p["sw"] - p["pl"]
                    max_val, max_idx = -np.inf, -1
                    for kh in range(p["hk"]):
                        for kw in range(p["wk"]):
                            cur_h, cur_w = h_start + kh, w_start + kw
                            if 0 <= cur_h < h and 0 <= cur_w < w:
                                val = x[i, cur_h, cur_w, ci] if mode == "nhwc" else x[i, ci, cur_h, cur_w]
                                if val > max_val or max_idx == -1:
                                    max_val = val
                                    if mode == "nhwc":
                                        max_idx = (cur_h * w + cur_w) * c + ci
                                    else:
                                        max_idx = (ci * h + cur_h) * w + cur_w
                    if mode == "nhwc":
                        y[i, hi, wi, ci] = max_val
                        argmax[i, hi, wi, ci] = max_idx
                    else:
                        y[i, ci, hi, wi] = max_val
                        argmax[i, ci, hi, wi] = max_idx
    return y, argmax


def max_pool_with_argmax_gen(x, p):
    if p["m"] == "nhwc":
        n, h, w, c = x.shape
    else:
        n, c, h, w = x.shape
    return compute_pool_core(x, p, n, c, h, w)


def gen_golden_data(case_name, in_dtype, idx_dtype):
    p = CASE_LIST.get(case_name)
    if p["m"] == "nhwc":
        shape = (p["n"], p["h"], p["w"], p["c"])
    else:
        shape = (p["n"], p["c"], p["h"], p["w"])

    x = np.random.uniform(-10, 10, size=shape).astype(in_dtype)
    y_g, arg_g = max_pool_with_argmax_gen(x, p)
    x.tofile("input.bin")
    y_g.tofile("y_golden.bin")
    arg_g.astype(idx_dtype).tofile("indices.bin")

DTYPE_MAP = {"float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64}

if __name__ == "__main__":
    gen_golden_data(sys.argv[1], DTYPE_MAP[sys.argv[2]], DTYPE_MAP[sys.argv[3]])