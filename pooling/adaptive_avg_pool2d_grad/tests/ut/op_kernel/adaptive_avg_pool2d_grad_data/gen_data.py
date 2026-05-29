#!/usr/bin/env python3
# coding: utf-8

import sys
import numpy as np

CASE_LIST = {
    # === BIG_KERNEL cases (TPL_BIG_KERNEL template, pre-computed tiling from original) ===
    "test_big_fp32_single_core_0": {"n": 1, "c": 8, "h_in": 8, "w_in": 8, "h_out": 4, "w_out": 4},
    "test_big_fp16_single_core_10": {"n": 1, "c": 8, "h_in": 8, "w_in": 8, "h_out": 4, "w_out": 4},
    "test_big_fp32_multi_core_20": {"n": 1, "c": 8, "h_in": 12, "w_in": 12, "h_out": 4, "w_out": 4},
    "test_big_fp32_large_nc_30": {"n": 1, "c": 256, "h_in": 4, "w_in": 4, "h_out": 2, "w_out": 2},
    # === SIMT cases (TPL_SIMT_KERNEL template) ===
    "test_simt_fp32_small":  {"n": 1, "c": 2, "h_in": 8, "w_in": 8, "h_out": 4, "w_out": 4},
    "test_simt_fp32_medium": {"n": 1, "c": 2, "h_in": 8, "w_in": 8, "h_out": 2, "w_out": 2},
}
DTYPE_MAP = {"float32": np.float32, "float16": np.float16}


def compute_grad(go, n, c, h_in, w_in, h_out, w_out):
    gi = np.zeros((n, c, h_in, w_in), dtype=go.dtype)
    for ni in range(n):
        for ci in range(c):
            for ho in range(h_out):
                ih_s = (ho * h_in) // h_out
                ih_e = ((ho + 1) * h_in + h_out - 1) // h_out
                for wo in range(w_out):
                    iw_s = (wo * w_in) // w_out
                    iw_e = ((wo + 1) * w_in + w_out - 1) // w_out
                    ks = (ih_e - ih_s) * (iw_e - iw_s)
                    val = go[ni, ci, ho, wo] / ks
                    for ih in range(ih_s, ih_e):
                        for iw in range(iw_s, iw_e):
                            gi[ni, ci, ih, iw] += val
    return gi


def gen_golden_data(case_name, in_dtype):
    p = CASE_LIST.get(case_name)
    shape_go = (p["n"], p["c"], p["h_out"], p["w_out"])
    go = np.random.uniform(-1.0, 1.0, size=shape_go).astype(in_dtype)
    gi = compute_grad(go, p["n"], p["c"], p["h_in"], p["w_in"], p["h_out"], p["w_out"])
    go.tofile("input_grad.bin")
    gi.tofile("x_grad_golden.bin")


if __name__ == "__main__":
    gen_golden_data(sys.argv[1], DTYPE_MAP[sys.argv[2]])
