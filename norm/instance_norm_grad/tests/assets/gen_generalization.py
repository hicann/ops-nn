#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Functional generalization case generator for InstanceNormGrad (arch35 / Ascend950).

Emits ~1000 TTK-kernel CSV rows covering the full TilingKey x dtype x shape matrix.
Column layout is copied verbatim from the smoke CSV
(norm/instance_norm_grad/tests/st/arch35/ttk_kernel_instance_norm_grad_st.csv).

Prototype (A2 input order): (dy, x, variance, mean, gamma) -> (pd_x, pd_gamma, pd_beta)
  layout NDHWC 5D  : dy/x/pd_x = [N, D, H, W, C]
  variance/mean    : [N, 1, 1, 1, C]  (size N*C, spatial dims collapsed)
  gamma/pd_*       : [C]
  dtype            : fp16 / fp32 only (NO bf16); all five inputs share one dtype.

TilingKeys (from op_kernel/instance_norm_grad.cpp + op_host/arch35/*tiling*):
  101 full_load  fp32 | 102 full_load  fp16
  301 recompute  fp32 | 302 recompute  fp16
  500 empty tensor (some axis == 0)

The host tiling (instance_norm_grad_tiling_arch35.cpp) picks full_load vs recompute by:
    rowBytes      = CeilAlign(cTile * tBytes, 32)
    fullLoadBytes = rowBytes * M * UB_COPIES_3(=3)
    reserveBytes  = PARAM_BUFFERS(=8) * CeilAlign(cTile, 8) * 4
    canUseUB      = (ubSize - reserveBytes) / DOUBLE_BUFFER(=2)
    full_load  iff fullLoadBytes <= canUseUB   else recompute
cTile == C when N >= coreNum; for small N it may shrink (C split across idle cores),
which only makes full_load *more* likely. We therefore use margins that hold for any
Ascend UB >= 192KiB and regardless of the C split:
    full_load  : M*C <= 4096      -> guaranteed key 101/102
    recompute  : M   >= 4096      -> guaranteed key 301/302 (even if cTile shrinks to 1,
                                     96*M > canUseUB for M >= 4096)
    empty      : N   == 0         -> xSize==0, gammaSize=C!=0 -> key 500.
                 (N=0 keeps m=D*H*W > 0 so the reference golden.py, which evaluates
                  2.0/m, stays crash-free; a *spatial* zero would divide by zero.)

Reproducibility: every random case draws from a per-case random.Random seeded by
hashlib.md5(stable_case_id) (NOT the process-unstable builtin hash()). Re-running the
script byte-for-byte reproduces the CSV.

Usage:
    python3 gen_generalization.py [out.csv]
The CSV is written to argv[1] (default ./generalization.csv). It is intentionally kept
OUT of the repo tree to keep the source checkout clean.
"""

import hashlib
import os
import sys
import random

# ---- TilingKey boundary knobs (see module docstring) -------------------------------------------
FULL_LOAD_MC_CAP = 4096  # M*C <= this  -> full_load  (safe for any ubSize >= 192KiB)
RECOMPUTE_M_MIN = 4096  # M   >= this  -> recompute  (safe even when cTile == 1)
MAX_TENSOR_ELEMS = (
    16_000_000  # cap N*M*C (dy/x element count) so cases stay runnable on HBM
)

# dtype -> (precision_tolerances, absolute_precision, full_load_key, recompute_key, short_tag)
DT = {
    "float32": ("((0.0001, 0.0001),)", "1.00E-05", 101, 301, "f32"),
    "float16": ("((0.001, 0.001),)", "1.00E-04", 102, 302, "f16"),
}

# C pools. "aligned" = multiple of 16 (32B for both fp16/fp32); fp32-aligned = multiple of 8
# (fp16 tail); tail = not 32B aligned for either dtype -> exercises DataCopyPad rightPadding.
C_ALIGNED = [16, 32, 64, 128, 256]
C_FP32_ALIGNED = [8, 24, 40, 72]
C_TAIL = [1, 3, 7, 17, 33, 63, 100, 127, 129, 255, 257]
C_POOL_FL = C_ALIGNED + [512] + C_FP32_ALIGNED + C_TAIL + C_TAIL  # tails weighted up
C_POOL_RC = [
    1,
    3,
    7,
    8,
    16,
    17,
    24,
    32,
    33,
    63,
    64,
    100,
    127,
    128,
]  # moderate C for recompute
C_EMPTY = [1, 8, 16, 17, 33, 64, 100, 128, 256]

HDR = (
    "testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,"
    "output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,"
    "output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,"
    "absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,"
    "soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries"
)


def q(s):
    return '"' + s + '"'


def t(dims):
    """Format an int tuple like Python repr: [8]->'(8,)', [1,2,3]->'(1, 2, 3)'."""
    if len(dims) == 1:
        return "(%d,)" % dims[0]
    return "(" + ", ".join(str(d) for d in dims) + ")"


def rng_for(cid):
    """Deterministic per-case RNG seeded from md5(case id) (not builtin hash())."""
    return random.Random(int(hashlib.md5(cid.encode()).hexdigest(), 16) % (2**32))


def build_row(name, dt, n, d, h, w, c, remark):
    x_shape = t([n, d, h, w, c])
    var_shape = t([n, 1, 1, 1, c])
    g_shape = t([c])
    in_shapes = "(%s, %s, %s, %s, %s)" % (
        x_shape,
        x_shape,
        var_shape,
        var_shape,
        g_shape,
    )
    out_shapes = "(%s, %s, %s)" % (x_shape, g_shape, g_shape)
    in_dtypes = "(" + ", ".join(["'%s'" % dt] * 5) + ")"
    out_dtypes = "(" + ", ".join(["'%s'" % dt] * 3) + ")"
    in_fmt = "(" + ", ".join(["'ND'"] * 5) + ")"
    out_fmt = "(" + ", ".join(["'ND'"] * 3) + ")"
    # per-input ranges: dy(-1,1) x(-1,1) variance(0.1,2 >=0) mean(-1,1) gamma(-1,1)
    ranges = "((-1, 1), (-1, 1), (0.1, 2), (-1, 1), (-1, 1))"
    tol, atol = DT[dt][0], DT[dt][1]
    fields = [
        name,
        "UNKNOWN",
        "instance_norm_grad",
        q(in_shapes),
        q(in_dtypes),
        q(in_fmt),
        q(out_shapes),
        q(out_dtypes),
        q(out_fmt),
        q(in_shapes),
        q(in_fmt),
        q(out_shapes),
        q(out_fmt),
        "{}",
        q(ranges),
        q(tol),
        atol,
        "()",
        "()",
        "TRUE",
        remark,
        "",
        "0",
        "",
        "()",
        "()",
    ]
    return ",".join(fields)


# ---- shape pickers -----------------------------------------------------------------------------
def pick_full_load_dhw(rng, c):
    """Pick D,H,W (small) so M=D*H*W and M*C <= FULL_LOAD_MC_CAP. Covers M=1 collapses."""
    cap_m = max(1, FULL_LOAD_MC_CAP // c)
    pool = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 8]
    for _ in range(16):
        d, h, w = rng.choice(pool), rng.choice(pool), rng.choice(pool)
        if 1 <= d * h * w <= cap_m:
            return d, h, w
    return 1, 1, rng.randint(1, cap_m)  # fallback: single spatial row


def pick_full_load(rng, c):
    d, h, w = pick_full_load_dhw(rng, c)
    m = d * h * w
    n_cap = max(1, MAX_TENSOR_ELEMS // (m * c))
    n_opts = [
        n for n in [1, 1, 1, 2, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128] if n <= n_cap
    ] or [1]
    return rng.choice(n_opts), d, h, w


def pick_recompute(rng, c):
    """Pick N,D,H,W with M=D*H*W >= RECOMPUTE_M_MIN and N*M*C <= MAX_TENSOR_ELEMS."""
    big_pool = [4096, 5000, 6144, 8192, 10240, 12288, 16384, 20480, 32768, 49152, 65536]
    for _ in range(24):
        big = rng.choice(big_pool)
        s1, s2 = rng.choice([1, 1, 2]), rng.choice([1, 1, 2])
        dims = [big, s1, s2]
        rng.shuffle(dims)
        d, h, w = dims
        m = d * h * w  # >= big >= RECOMPUTE_M_MIN
        if (
            m * c > MAX_TENSOR_ELEMS
        ):  # even N=1 would blow the HBM budget -> resample smaller M
            continue
        n_cap = max(1, MAX_TENSOR_ELEMS // (m * c))
        n_opts = [n for n in [1, 1, 2, 2, 3, 4, 6, 8] if n <= n_cap]
        if not n_opts:
            continue
        return rng.choice(n_opts), d, h, w
    return 1, 1, 1, RECOMPUTE_M_MIN  # fallback


def name_of(cid, key, dt, n, d, h, w, c):
    return "%s_k%d_%s_N%d_D%d_H%d_W%d_C%d" % (cid, key, DT[dt][4], n, d, h, w, c)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "generalization.csv"
    rows = [HDR]
    dist = {101: 0, 102: 0, 301: 0, 302: 0, 500: 0}

    # ---------- 1) random main body (seeded, 4-way balanced over key x dtype) ----------
    RANDOM_TOTAL = 880
    for i in range(RANDOM_TOTAL):
        b = i % 4
        mode, dt = (
            ("fl", "float32"),
            ("fl", "float16"),
            ("rc", "float32"),
            ("rc", "float16"),
        )[b]
        cid = "ing_rand_%04d" % i
        rng = rng_for(cid)
        if mode == "fl":
            c = rng.choice(C_POOL_FL)
            n, d, h, w = pick_full_load(rng, c)
            key = DT[dt][2]
        else:
            c = rng.choice(C_POOL_RC)
            n, d, h, w = pick_recompute(rng, c)
            key = DT[dt][3]
        rows.append(
            build_row(
                name_of(cid, key, dt, n, d, h, w, c),
                dt,
                n,
                d,
                h,
                w,
                c,
                "key%d_%s_%s_rand" % (key, mode, DT[dt][4]),
            )
        )
        dist[key] += 1

    # ---------- 2) system-boundary cases (explicit, both dtypes) ----------
    # (mode, N, D, H, W, C, note); expanded over both dtypes.
    BND = [
        # -- full_load extremes --
        ("fl", 1, 1, 1, 1, 1, "min_single_core_M1_C1"),
        ("fl", 1, 1, 1, 1, 8, "M1_C8"),
        ("fl", 1, 1, 1, 1, 16, "M1_C16"),
        ("fl", 1, 1, 1, 1, 17, "M1_tailC17"),
        ("fl", 1, 1, 1, 1, 64, "M1_C64"),
        ("fl", 2, 1, 1, 1, 32, "M1_N2"),
        ("fl", 1, 1, 1, 7, 16, "singleRowW7"),
        ("fl", 1, 1, 3, 1, 33, "singleRowH3_tailC"),
        # aligned vs unaligned C pairs (M=4)
        ("fl", 2, 1, 2, 2, 16, "pair_alignedC16"),
        ("fl", 2, 1, 2, 2, 17, "pair_tailC17"),
        ("fl", 2, 1, 2, 2, 64, "pair_alignedC64"),
        ("fl", 2, 1, 2, 2, 63, "pair_tailC63"),
        ("fl", 2, 1, 2, 2, 128, "pair_alignedC128"),
        ("fl", 2, 1, 2, 2, 127, "pair_tailC127"),
        ("fl", 2, 1, 2, 2, 256, "pair_alignedC256"),
        ("fl", 2, 1, 2, 2, 257, "pair_tailC257"),
        # multi-core (large N -> N>=coreNum, cTile=C)
        ("fl", 64, 1, 2, 2, 64, "multicore_N64"),
        ("fl", 128, 1, 1, 2, 16, "multicore_N128"),
        ("fl", 96, 1, 2, 2, 8, "multicore_N96_C8"),
        # small N large C -> C split across cores (peak large-C full_load)
        ("fl", 1, 1, 2, 4, 512, "topC512_Csplit"),
        ("fl", 1, 1, 4, 4, 256, "topC256_Csplit"),
        ("fl", 1, 1, 1, 8, 512, "topC512_M8"),
        # -- recompute extremes --
        ("rc", 1, 1, 1, 4096, 1, "min_single_core_C1_M4096"),
        ("rc", 1, 1, 1, 4096, 8, "min_recompute_C8"),
        ("rc", 1, 1, 1, 4096, 16, "min_recompute_C16"),
        ("rc", 1, 1, 1, 4096, 17, "min_recompute_tailC17"),
        # cross-N reduction (N>1) recompute
        ("rc", 4, 1, 1, 4096, 32, "crossN_N4"),
        ("rc", 8, 1, 2, 4096, 16, "crossN_N8"),
        ("rc", 2, 1, 1, 8192, 33, "crossN_N2_tailC33"),
        # aligned vs unaligned C pairs, recompute
        ("rc", 2, 1, 1, 8192, 64, "rc_pair_alignedC64"),
        ("rc", 2, 1, 1, 8192, 63, "rc_pair_tailC63"),
        ("rc", 2, 1, 1, 8192, 128, "rc_pair_alignedC128"),
        ("rc", 2, 1, 1, 8192, 127, "rc_pair_tailC127"),
        # big C recompute (C split + M tiling)
        ("rc", 2, 1, 1, 8192, 256, "rc_bigC256"),
        ("rc", 1, 1, 1, 16384, 256, "rc_bigC256_deepM"),
        # deep M-tile loop (top of tile-loop depth)
        ("rc", 1, 1, 1, 131072, 16, "top_deepLoop_M131072"),
        ("rc", 1, 1, 1, 262144, 8, "top_deepLoop_M262144"),
        ("rc", 1, 1, 1, 524288, 8, "top_deepLoop_M524288"),
        ("rc", 1, 2, 2, 32768, 64, "top_3d_bigM"),
    ]
    bidx = 0
    for dt in ("float32", "float16"):
        for mode, n, d, h, w, c, note in BND:
            if n * d * h * w * c > MAX_TENSOR_ELEMS:
                continue
            key = DT[dt][2] if mode == "fl" else DT[dt][3]
            cid = "ing_bnd_%04d" % bidx
            bidx += 1
            rows.append(
                build_row(
                    name_of(cid, key, dt, n, d, h, w, c),
                    dt,
                    n,
                    d,
                    h,
                    w,
                    c,
                    "key%d_%s_%s_bnd_%s" % (key, mode, DT[dt][4], note),
                )
            )
            dist[key] += 1

    # ---------- 3) empty-tensor cases -> key 500 (N=0, golden-safe) ----------
    eidx = 0
    for dt in ("float32", "float16"):
        for c in C_EMPTY:
            for d, h, w in ((1, 1, 1), (1, 2, 2), (2, 3, 4)):
                cid = "ing_empty_%04d" % eidx
                eidx += 1
                name = "%s_k500_%s_N0_D%d_H%d_W%d_C%d" % (cid, DT[dt][4], d, h, w, c)
                rows.append(
                    build_row(name, dt, 0, d, h, w, c, "key500_empty_%s_N0" % DT[dt][4])
                )
                dist[500] += 1

    out_dir = os.path.dirname(os.path.abspath(out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(rows) + "\n")

    total = len(rows) - 1
    print("generated %d cases -> %s" % (total, out))
    print("TilingKey distribution:")
    for k in (101, 102, 301, 302, 500):
        label = {
            101: "full_load fp32",
            102: "full_load fp16",
            301: "recompute fp32",
            302: "recompute fp16",
            500: "empty (N=0)",
        }[k]
        print("  key %d  %-16s : %4d" % (k, label, dist[k]))


if __name__ == "__main__":
    main()
