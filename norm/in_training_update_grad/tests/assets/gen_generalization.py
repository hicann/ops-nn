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
Functional generalization case generator for arch35 op INTrainingUpdateGrad (TTK kernel CSV).

Prototype (no attributes):
    (dy, x, variance, mean) -> (res_gamma, res_beta)
    dy / x        : float16 or float32 (same dtype), full spatial dims
    variance/mean : float32 always, spatial dims = 1
    res_gamma/res_beta : float32 always, spatial dims = 1
    format        : NDC1HWC0 (6D)  [N, D, C1, H, W, C0], C0 = 16
    reduce axes   : D, H, W ; kept axes : N, C1, C0
    math          : x_norm = (x - mean) * rsqrt(variance + 1e-6)
                    res_gamma = sum_{D,H,W}(dy * x_norm) ; res_beta = sum_{D,H,W}(dy)   (keepdims)

TilingKey routing (mirrors op_host/arch35/in_training_update_grad_tiling_arch35.cpp exactly):
    50000  ReduceEmpty : R == 0   (any of D/H/W == 0) ; both outputs = 0.0
    100000 FullLoad    : R > 0 AND D <= 65535 AND (one group's R*C0 block fits UB, double buffered)
    200000 Stream      : R > 0 AND (does not fit UB OR D > 65535)

The FullLoad UB-fit predicate is copied byte-for-byte from InTrainingUpdateGradFullLoadTiling::IsCapable
(Ascend950: UB_SIZE = 253952, vectorLength/regPad = 256, C0 = 16), giving the boundaries:
    fp32 : R = D*H*W <= 984  -> FullLoad ; R >= 985  -> Stream
    fp16 : R = D*H*W <= 1968 -> FullLoad ; R >= 1969 -> Stream

Only shape / CSV are produced here; nothing is run on NPU/TTK, no operator code is touched.
"""

import csv
import hashlib
import math
import os
import random
import sys

# ---------------------------------------------------------------------------
# Platform / tiling constants (Ascend950, arch35) -- keep in sync with the host tiling.
# ---------------------------------------------------------------------------
UB_SIZE = 253952  # bytes, Ascend950 UB
REG_PAD = 256  # vectorLength (VReg width in bytes) on arch35
C0 = 16  # NDC1HWC0 last dim
FP32_BYTE = 4
FP16_BYTE = 2
DOUBLE_BUFFER = 2
MAX_BLOCK_COUNT = (
    65535  # DataCopyExtParams.blockCount is uint16 -> D must be <= this for FullLoad
)

KEY_REDUCE_EMPTY = 50000
KEY_FULL_LOAD = 100000
KEY_STREAM = 200000

# FullLoad R upper bounds derived from full_load_fits() below (documentation only; predicate is authoritative).
RMAX = {"float32": 984, "float16": 1968}

MEM_CAP = 48 * 1024 * 1024  # cap dy+x host bytes per case so the CSV stays cheap to run

# Master RNG seed -> the whole generation is deterministic / reproducible run-to-run.
MASTER_SEED = 0x1704A26D

DEFAULT_OUT = (
    "/tmp/claude-0/-workspace/ff64f473-fb09-46f9-94c6-794b8bca8052/"
    "scratchpad/gen/in_training_update_grad/generalization.csv"
)

CSV_HEADER = [
    "testcase_name",
    "network_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes",
    "output_shape_unknown_indexes",
    "is_enabled",
    "remark",
    "soc_series",
    "priority",
    "dump_file_prefix",
    "manual_input_binaries",
    "manual_golden_binaries",
]

DT_TAG = {"float32": "f32", "float16": "f16"}


# ---------------------------------------------------------------------------
# Tiling-key classification -- exact port of the host IsCapable() chain.
# ---------------------------------------------------------------------------
def full_load_fits(reduce_r, dtype):
    elem = FP16_BYTE if dtype == "float16" else FP32_BYTE
    spatial_bytes = reduce_r * C0 * elem
    need = (
        DOUBLE_BUFFER * (spatial_bytes + REG_PAD) * 2  # dy + x
        + 2 * (C0 * FP32_BYTE + REG_PAD)  # variance + mean
        + DOUBLE_BUFFER * (C0 * FP32_BYTE) * 2  # res_gamma + res_beta
        + C0 * FP32_BYTE
    )  # rstd
    return need <= UB_SIZE


def classify(n, d, c1, h, w, dtype):
    reduce_r = d * h * w
    if reduce_r == 0:
        return KEY_REDUCE_EMPTY
    if d <= MAX_BLOCK_COUNT and full_load_fits(reduce_r, dtype):
        return KEY_FULL_LOAD
    return KEY_STREAM


REGIME_TAG = {
    KEY_REDUCE_EMPTY: "reduce_empty",
    KEY_FULL_LOAD: "full_load",
    KEY_STREAM: "stream",
}


# ---------------------------------------------------------------------------
# Data-range pools (variance range is always >= 0). Chosen per-case from md5(name).
# ---------------------------------------------------------------------------
DY_RANGES = [(-1, 1), (-2, 2), (-0.5, 0.5), (-4, 4), (-10, 10), (-1, 1), (-3, 3)]
X_RANGES = [(-1, 1), (-2, 2), (-3, 3), (-1, 1), (-5, 5), (-0.5, 0.5)]
VAR_RANGES = [
    (0.1, 2),
    (0.01, 1),
    (0.5, 5),
    (0.001, 0.1),
    (0.5, 10),
    (0.1, 2),
    (0, 1),
]  # all >= 0
MEAN_RANGES = [(-1, 1), (0, 2), (-5, 5), (-0.5, 0.5), (-2, 2), (-1, 1)]


def pick_ranges(name):
    """Deterministic per-case data ranges seeded by md5(testcase_name)."""
    seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    r = random.Random(seed)
    return (
        r.choice(DY_RANGES),
        r.choice(X_RANGES),
        r.choice(VAR_RANGES),
        r.choice(MEAN_RANGES),
    )


def tolerances(dtype):
    if dtype == "float32":
        return ((1e-4, 1e-4),), 1e-5
    return ((1e-3, 1e-3),), 1e-4


# ---------------------------------------------------------------------------
# CSV row assembly.
# ---------------------------------------------------------------------------
def build_row(name, n, d, c1, h, w, dtype):
    key = classify(n, d, c1, h, w, dtype)
    reduce_r = d * h * w
    group_num = n * c1

    dy_shape = [n, d, c1, h, w, C0]
    stat_shape = [n, 1, c1, 1, 1, C0]  # variance / mean / outputs (spatial dims -> 1)
    in_shapes = [dy_shape, dy_shape, stat_shape, stat_shape]
    out_shapes = [stat_shape, stat_shape]

    in_dtypes = [dtype, dtype, "float32", "float32"]
    in_formats = ["NDC1HWC0"] * 4
    out_dtypes = ["float32", "float32"]
    out_formats = ["NDC1HWC0", "NDC1HWC0"]

    ptol, atol = tolerances(dtype)
    ranges = pick_ranges(name)
    remark = "key{}_{}_{}_g{}_R{}_D{}".format(
        key, REGIME_TAG[key], DT_TAG[dtype], group_num, reduce_r, d
    )

    return [
        name,
        "UNKNOWN",
        "in_training_update_grad",
        str(in_shapes),
        str(in_dtypes),
        str(in_formats),
        str(out_shapes),
        str(out_dtypes),
        str(out_formats),
        str(in_shapes),
        str(in_formats),
        str(out_shapes),
        str(out_formats),
        "{}",
        str(ranges),
        str(ptol),
        str(atol),
        "()",
        "()",
        "1",
        remark,
        "",
        "",
        "",
        "()",
        "()",
    ], key


# ---------------------------------------------------------------------------
# Memory guard: shrink group count (C1 then N) so dy+x host bytes stay under MEM_CAP.
# N/C1 do not affect the tiling key, so this never changes the key.
# ---------------------------------------------------------------------------
def mem_guard(n, c1, reduce_r, dtype):
    elem = FP16_BYTE if dtype == "float16" else FP32_BYTE

    def nbytes(nn, cc):
        return nn * cc * reduce_r * C0 * elem * 2  # dy + x

    if nbytes(n, c1) <= MEM_CAP:
        return n, c1
    if nbytes(n, 1) <= MEM_CAP:
        return n, 1
    return 1, 1


# ---------------------------------------------------------------------------
# Systematic (boundary / DFX / core-regime) cases.  (tag, N, D, C1, H, W) emitted for both dtypes,
# unless the tuple is in DTYPE_SPECIFIC (emitted for one dtype only, to sit exactly on that dtype's boundary).
# ---------------------------------------------------------------------------
SYS_BOTH = [
    # ---- FullLoad: single core / spatial-shape variety ----
    ("fl_min", 1, 1, 1, 1, 1),  # R=1  minimal single row, single core
    ("fl_tiny", 1, 1, 1, 2, 2),  # R=4
    ("fl_row_1vl", 1, 1, 1, 8, 8),  # R=64 == one fp32 VL
    ("fl_cross_vl", 1, 1, 1, 16, 16),  # R=256 spans several VLs
    ("fl_d3", 1, 3, 1, 4, 4),  # R=48  D-strided aggregation (D>1)
    ("fl_d5_c1", 1, 5, 3, 4, 4),  # R=80  D>1 & C1>1
    ("fl_aspect_h1", 1, 1, 1, 1, 200),  # R=200 extreme aspect H=1 (wide)
    ("fl_aspect_w1", 1, 1, 1, 200, 1),  # R=200 extreme aspect W=1 (tall)
    (
        "fl_dmed",
        1,
        8,
        1,
        10,
        10,
    ),  # R=800 larger D, near-ish fp32 boundary, still FullLoad both
    # ---- FullLoad: multi-core regimes (groupNum = N*C1) ----
    ("fl_multi8", 2, 1, 4, 4, 4),  # groupNum=8   R=16
    ("fl_multi64", 8, 1, 8, 3, 3),  # groupNum=64  (== AIV count) R=9
    ("fl_tail65", 65, 1, 1, 2, 2),  # groupNum=65  1-elem tail over 64 cores
    ("fl_multi128", 64, 1, 2, 2, 2),  # groupNum=128 perCoreGroups=2, all 64 cores
    ("fl_tail200", 25, 1, 8, 2, 2),  # groupNum=200 perCoreGroups=4 tail
    ("fl_big_groups", 32, 1, 16, 2, 2),  # groupNum=512 heavy oversubscription
    ("fl_n_c1_d", 4, 2, 3, 5, 5),  # N=4,D=2,C1=3 mixed, R=50
    # ---- Stream: large spatial / large D (both dtypes) ----
    ("st_big_spatial", 1, 2, 1, 32, 32),  # R=2048  (smoke parity)
    ("st_bigR", 1, 1, 1, 100, 100),  # R=10000
    ("st_bigR_d", 1, 4, 1, 50, 50),  # R=10000 with D-strided aggregation
    ("st_aspect_wide", 1, 1, 1, 1, 5000),  # R=5000  H=1 wide stream
    ("st_aspect_tall", 1, 1, 1, 5000, 1),  # R=5000  W=1 tall stream
    ("st_multi", 4, 2, 2, 40, 40),  # groupNum=8 stream, R=3200
    ("st_dmed", 1, 1000, 1, 2, 2),  # R=4000 medium-large D
    (
        "st_bigD_blockcount",
        1,
        70000,
        1,
        1,
        1,
    ),  # D>65535 -> Stream via blockCount limit (R=70000)
    ("st_bigD_66000", 1, 66000, 1, 1, 1),  # just above the 65535 boundary
    # ---- ReduceEmpty (R == 0): D/H/W == 0 variants ----
    ("re_d0", 2, 0, 3, 4, 4),  # D=0
    ("re_h0", 2, 1, 3, 0, 4),  # H=0
    ("re_w0", 2, 1, 3, 4, 0),  # W=0
    ("re_single", 1, 0, 1, 2, 2),  # groupNum=1 empty
    ("re_multi", 32, 1, 8, 0, 4),  # groupNum=256 multi-core empty (H=0)
    ("re_hw0", 1, 1, 1, 0, 0),  # H=0 & W=0
    ("re_bignc1", 16, 0, 16, 3, 3),  # groupNum=256 empty (D=0)
]

# (tag, N, D, C1, H, W, dtype) -- sit exactly on that dtype's FullLoad/Stream boundary.
SYS_DTYPE_SPECIFIC = [
    ("fl_boundary_max", 1, 1, 1, 24, 41, "float32"),  # R=984  -> FullLoad (fp32 max)
    ("st_boundary_over", 1, 1, 1, 5, 197, "float32"),  # R=985  -> Stream  (fp32 min)
    ("fl_boundary_max", 1, 1, 1, 48, 41, "float16"),  # R=1968 -> FullLoad (fp16 max)
    ("st_boundary_over", 1, 1, 1, 11, 179, "float16"),  # R=1969 -> Stream  (fp16 min)
]


# ---------------------------------------------------------------------------
# Random-body samplers per regime (biased so each accepted sample lands in the intended key bucket).
# ---------------------------------------------------------------------------
N_MENU = [1, 1, 1, 2, 2, 3, 4, 4, 6, 8, 8, 12, 16, 24, 32, 48, 64]
C1_MENU = [1, 1, 1, 2, 2, 3, 4, 4, 6, 8, 16]
FULL_D_MENU = [1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8]
STREAM_N_MENU = [1, 1, 2, 2, 3]
STREAM_C1_MENU = [1, 1, 2]
BIGD_MENU = [65536, 66000, 70000, 80000, 100000, 131072]
EMPTY_DIM_MENU = [1, 2, 3, 4, 5, 8, 16, 32]


def sample_full(rng, dtype):
    rmax = RMAX[dtype]
    d = rng.choice(FULL_D_MENU)
    budget = max(1, rmax // d)
    h = rng.randint(1, min(budget, 96))
    wmax = max(1, budget // h)
    w = rng.randint(1, wmax)
    n = rng.choice(N_MENU)
    c1 = rng.choice(C1_MENU)
    return n, d, c1, h, w


def sample_stream(rng, dtype):
    if rng.random() < 0.22:  # big-D route (blockCount limit)
        d = rng.choice(BIGD_MENU)
        return rng.choice([1, 1, 2]), d, 1, 1, 1
    rmax = RMAX[dtype]
    d = rng.choice([1, 2, 3, 4, 5, 8])
    base = rmax // d + 1  # ensures d*base > rmax (floor-div property)
    hw = base + rng.randint(0, 6000)
    h = rng.randint(1, max(1, math.isqrt(hw)))
    w = -(-hw // h)  # ceil -> h*w >= hw  => d*h*w > rmax
    n = rng.choice(STREAM_N_MENU)
    c1 = rng.choice(STREAM_C1_MENU)
    return n, d, c1, h, w


def sample_empty(rng, dtype):
    which = rng.choice(["D", "H", "W"])
    d = 0 if which == "D" else rng.choice(EMPTY_DIM_MENU)
    h = 0 if which == "H" else rng.choice(EMPTY_DIM_MENU)
    w = 0 if which == "W" else rng.choice(EMPTY_DIM_MENU)
    return rng.choice(N_MENU), d, rng.choice(C1_MENU), h, w


SAMPLERS = {
    KEY_FULL_LOAD: sample_full,
    KEY_STREAM: sample_stream,
    KEY_REDUCE_EMPTY: sample_empty,
}

# random-body targets per (key, dtype)
RAND_TARGETS = {
    KEY_FULL_LOAD: 230,
    KEY_STREAM: 155,
    KEY_REDUCE_EMPTY: 85,
}


def gen():
    rng = random.Random(MASTER_SEED)
    rows = []
    seen_shapes = set()  # (n, d, c1, h, w, dtype) -> shape diversity
    idx = 0

    def add(tag, n, d, c1, h, w, dtype):
        nonlocal idx
        name = "in_training_update_grad_gen{:04d}_{}_{}".format(idx, tag, DT_TAG[dtype])
        row, key = build_row(name, n, d, c1, h, w, dtype)
        rows.append((key, dtype, row))
        seen_shapes.add((n, d, c1, h, w, dtype))
        idx += 1
        return key

    # ---- systematic cases ----
    for tag, n, d, c1, h, w in SYS_BOTH:
        for dtype in ("float32", "float16"):
            n2, c12 = mem_guard(n, c1, d * h * w, dtype)
            add(tag, n2, d, c12, h, w, dtype)
    for tag, n, d, c1, h, w, dtype in SYS_DTYPE_SPECIFIC:
        n2, c12 = mem_guard(n, c1, d * h * w, dtype)
        add(tag, n2, d, c12, h, w, dtype)

    # ---- random body: fill each (key, dtype) bucket ----
    for dtype in ("float32", "float16"):
        for key, target in RAND_TARGETS.items():
            sampler = SAMPLERS[key]
            got = 0
            attempts = 0
            while got < target and attempts < target * 200:
                attempts += 1
                n, d, c1, h, w = sampler(rng, dtype)
                if classify(n, d, c1, h, w, dtype) != key:
                    continue
                n, c1 = mem_guard(n, c1, d * h * w, dtype)
                shp = (n, d, c1, h, w, dtype)
                if shp in seen_shapes and rng.random() < 0.9:
                    continue  # prefer shape diversity, but don't loop forever
                add(REGIME_TAG[key], n, d, c1, h, w, dtype)
                got += 1

    return rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = gen()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for _key, _dt, row in rows:
            writer.writerow(row)

    # ---- distribution report ----
    from collections import Counter

    by_key = Counter(k for k, _dt, _r in rows)
    by_dt = Counter(dt for _k, dt, _r in rows)
    by_key_dt = Counter((k, dt) for k, dt, _r in rows)

    print("CSV written: {}".format(out_path))
    print("total cases: {}".format(len(rows)))
    print("\nby TilingKey:")
    for k in (KEY_REDUCE_EMPTY, KEY_FULL_LOAD, KEY_STREAM):
        print("  {:6d} {:<13s}: {}".format(k, REGIME_TAG[k], by_key.get(k, 0)))
    print("\nby dtype:")
    for dt in ("float32", "float16"):
        print("  {:<8s}: {}".format(dt, by_dt.get(dt, 0)))
    print("\nby (TilingKey, dtype):")
    for k in (KEY_REDUCE_EMPTY, KEY_FULL_LOAD, KEY_STREAM):
        for dt in ("float32", "float16"):
            print(
                "  {:6d} {:<13s} {:<8s}: {}".format(
                    k, REGIME_TAG[k], dt, by_key_dt.get((k, dt), 0)
                )
            )
    # sanity: every key must be non-empty
    assert all(
        by_key.get(k, 0) > 0 for k in (KEY_REDUCE_EMPTY, KEY_FULL_LOAD, KEY_STREAM)
    ), "a TilingKey has no cases!"


if __name__ == "__main__":
    main()
