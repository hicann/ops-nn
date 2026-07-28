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
#
# Generate the L2NormalizeGrad arch35 (Ascend950) functional-generalization TTK kernel case set.
#
# Prototype (arch35, aligned to A2 910B/910C support face):
#   inputs  : x, y, dy   -- same shape, same dtype, format ND
#   output  : dx         -- same shape/dtype as x
#   dtype   : float16 / float32  (NO bf16)
#   attrs   : dim (ListInt, default [1], single normalization axis), eps (Float, default 1e-4)
#
# TilingKey coverage (routing replicated from op_host/arch35/l2_normalize_grad_tiling.cpp
# SelectTemplate; [outer, D, inner] = split of x-shape around the resolved `dim`):
#   8000  Empty     : totalNum == 0 (any dim is 0)
#   7000  FullLoad  : inner == 1 and D <= DX_UB_FACTOR (6144)
#   7010  SplitD    : inner == 1 and D  > DX_UB_FACTOR (6144)
#   7020  Strided   : inner  > 1 (dim on a middle axis, e.g. 4D NCHW dim=1)
#
# Golden is computed live by tests/assets/golden.py (closed form == kernel math), so the harness
# generates x/y/dy independently in (-1,1); golden and kernel both consume the provided y directly.
# absolute_precision provides a small absolute floor so near-zero dx (dy - y*s cancels through 0)
# does not false-fail on the relative criterion; rtol is per-dtype (fp16 1e-3, fp32 1e-4).
#
# Reproducible: each random case draws its shape/dtype/attrs from a per-case RNG seeded by
# md5(stable case key), so the set is stable and order-independent.
#
# Usage: python3 gen_generalization.py [out.csv]
#   default out = <this dir>/generalization.csv

import csv
import hashlib
import os
import sys

OP_NAME = "l2_normalize_grad"
DX_UB_FACTOR = 6144  # full-load threshold (fp32 elements), matches tiling + kernel
MAX_NUMEL = 2_000_000  # per-tensor element cap to keep case data bounded
DEFAULT_EPS = 0.0001  # proto/contract default (1e-4)

# dtype -> (rtol/precision_tolerances value, absolute_precision string)
DTYPES = {
    "float32": (0.0001, "1e-5"),
    "float16": (0.001, "1e-3"),
}
TAG = {"float32": "f32", "float16": "f16"}

# eps variety (mostly default; a slice exercises the clamp/eps path). golden is the closed form so
# any eps stays consistent between golden and kernel.
EPS_POOL = [1e-12, 1e-6, 1e-5, 1e-2, 1.0]

HDR = [
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


def rng_for(key):
    """Deterministic per-case RNG seeded from md5(key) (reproducible, order-independent)."""
    import random

    seed = int(hashlib.md5(str(key).encode()).hexdigest(), 16) & 0xFFFFFFFF
    return random.Random(seed)


def resolve_dim(dim, rank):
    d = dim + rank if dim < 0 else dim
    return d


def classify(shape, dim):
    """Replicates op_host SelectTemplate: returns (tilingkey, outer, D, inner)."""
    rank = len(shape)
    d = resolve_dim(dim, rank)
    total = 1
    for s in shape:
        total *= s
    if total == 0:
        return 8000, 0, 0, 0
    outer = 1
    for i in range(d):
        outer *= shape[i]
    dlen = shape[d]
    inner = 1
    for i in range(d + 1, rank):
        inner *= shape[i]
    if inner == 1:
        key = 7000 if dlen <= DX_UB_FACTOR else 7010
    else:
        key = 7020
    return key, outer, dlen, inner


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def make_row(name, dtype, shape, dim, eps, remark):
    rtol, atol = DTYPES[dtype]
    shp = tuple(int(s) for s in shape)
    in_shapes = (shp, shp, shp)
    row = {
        "testcase_name": name,
        "network_name": "UNKNOWN",
        "op_name": OP_NAME,
        "input_shapes": repr(in_shapes),
        "input_dtypes": repr((dtype, dtype, dtype)),
        "input_formats": repr(("ND", "ND", "ND")),
        "output_shapes": repr((shp,)),
        "output_dtypes": repr((dtype,)),
        "output_formats": repr(("ND",)),
        "input_ori_shapes": repr(in_shapes),
        "input_ori_formats": repr(("ND", "ND", "ND")),
        "output_ori_shapes": "",
        "output_ori_formats": "",
        "attributes": repr({"dim": [int(dim)], "eps": eps}),
        "input_data_ranges": repr([[-1, 1], [-1, 1], [-1, 1]]),
        "precision_tolerances": repr(((rtol, rtol),)),
        "absolute_precision": atol,
        "output_inplace_indexes": "()",
        "output_shape_unknown_indexes": "()",
        "is_enabled": "1",
        "remark": remark,
        "soc_series": "",
        "priority": "",
        "dump_file_prefix": "",
        "manual_input_binaries": "()",
        "manual_golden_binaries": "()",
    }
    return [row[c] for c in HDR]


def eps_str(eps):
    return "e" + repr(eps).replace("-", "m").replace(".", "p").replace("+", "")


def add_case(rows, seen, dtype, shape, dim, eps, prefix):
    key, outer, dlen, inner = classify(shape, dim)
    shp_s = "x".join(str(s) for s in shape)
    base = (
        f"{OP_NAME}_{prefix}_{TAG[dtype]}_k{key}_{shp_s}_d{str(dim).replace('-', 'm')}"
    )
    name = base
    n = 1
    while name in seen:  # guard against collisions
        n += 1
        name = f"{base}_{n}"
    seen.add(name)
    remark = f"key{key} {dtype} shape[{shp_s}] dim={dim} eps={eps} outer={outer} D={dlen} inner={inner}"
    rows.append(make_row(name, dtype, shape, dim, eps, remark))
    return key


# ---------------------------------------------------------------------------
# Part A: systematic boundary cases (deterministic, both dtypes)
# ---------------------------------------------------------------------------
def build_systematic(rows, seen):
    dts = list(DTYPES)

    # A1  7000 full_load (inner==1): aligned vs non-aligned D pairs, single/multi-core, tail blocks.
    #     (outer, D, dim)
    full_specs = [
        (1, 1, 1),  # single element / single group / single core
        (1, 8, 1),  # single row, aligned
        (1, 17, 1),  # single row, non-aligned tail
        (7, 64, 1),  # sub-core, aligned
        (32, 128, 1),  # aligned
        (48, 100, -1),  # non-aligned tail (smoke), dim=-1
        (65, 127, 1),  # multi-core tail + non-aligned D
        (100, 33, -1),  # small non-aligned tail (smoke)
        (256, 256, 1),  # aligned mid
        (512, 511, 1),  # non-aligned, multi-core
        (1000, 1024, -1),  # large multi-core, aligned
        (13, 6144, 1),  # D == threshold, still 7000
        (16, 6143, 1),  # just under threshold, non-aligned
        (2048, 32, 1),  # many rows, tiny D (deep multi-core)
        (33, 4096, -1),  # aligned large D
        (128, 257, 1),  # non-aligned prime-ish D
    ]
    for dt in dts:
        for outer, d, dim in full_specs:
            add_case(rows, seen, dt, (outer, d), dim, DEFAULT_EPS, "bd")

    # A2  7010 split_d (inner==1, D>6144): threshold+1, pow2/non-pow2, probe-the-ceiling large D.
    split_specs = [
        (1, 6145, 1),  # just over threshold, single group/core
        (8, 8192, 1),  # pow2 (smoke fp16)
        (8, 10000, 1),  # non-pow2 (smoke fp32)
        (16, 8192, -1),  # pow2, multi-core, dim=-1
        (4, 12288, 1),  # larger pow2-ish
        (2, 16384, 1),  # ceiling probe: very large D
        (32, 7000, 1),  # multi-core, moderate over threshold
        (64, 6400, -1),  # multi-core, just over
        (3, 9973, 1),  # prime D
        (1, 6145, -1),  # threshold+1, dim=-1
    ]
    for dt in dts:
        for outer, d, dim in split_specs:
            add_case(rows, seen, dt, (outer, d), dim, DEFAULT_EPS, "bd")

    # A3  7020 strided (inner>1): dim on a middle axis, various inner (aligned/non-aligned), dim=-2.
    strided_specs = [
        ((4, 8, 16, 16), 1),  # smoke NCHW dim=1: outer4 D8 inner256
        ((2, 3, 4), 1),  # tiny 3D
        ((8, 16, 32), 1),  # 3D
        ((4, 4, 4, 4), 1),  # 4D inner=16
        ((4, 4, 4, 4), 2),  # 4D dim=2 inner=4
        ((2, 64, 7, 7), 1),  # NCHW non-aligned inner=49
        ((1, 3, 224, 224), 1),  # image-like, big inner=50176
        ((2, 32, 17), 1),  # inner=17 non-aligned
        ((3, 5, 8, 9), 2),  # dim=2 middle: outer15 D8 inner9
        ((2, 3, 4, 5), -2),  # negative middle dim=-2 -> 2
        ((16, 16, 16), 1),  # cube
        ((4, 8, 3, 3), 1),  # inner=9
    ]
    for dt in dts:
        for shape, dim in strided_specs:
            add_case(rows, seen, dt, shape, dim, DEFAULT_EPS, "bd")

    # A4  8000 empty: a zero dim in outer / D / inner positions, several ranks.
    empty_specs = [
        ((0, 4), 1),
        ((4, 0), 1),
        ((0,), 0),
        ((3, 0, 5), 1),
        ((0, 8, 16), 1),
        ((2, 3, 0), 2),
        ((5, 0), -1),
        ((2, 0, 4, 4), 1),
        ((0, 3, 4, 5), 1),
    ]
    for dt in dts:
        for shape, dim in empty_specs:
            add_case(rows, seen, dt, shape, dim, DEFAULT_EPS, "bd")

    # A5  eps variety on a fixed 7000 shape (exercise the eps/clamp path).
    for dt in dts:
        for eps in EPS_POOL:
            add_case(rows, seen, dt, (64, 256), 1, eps, "bd")

    # A6  rank/last-axis variety (inner==1 via dim=last / 1D).
    rank_specs = [
        ((128,), 0),  # 1D, dim=0
        ((129,), -1),  # 1D, dim=-1, non-aligned
        ((3, 4, 5), 2),  # 3D last axis
        ((3, 4, 5), -1),  # 3D dim=-1
        ((2, 3, 4, 5), 3),  # 4D last axis
        ((2, 3, 4, 5), -1),  # 4D dim=-1
        ((7, 11, 13), 2),  # 3D prime dims, last axis
    ]
    for dt in dts:
        for shape, dim in rank_specs:
            add_case(rows, seen, dt, shape, dim, DEFAULT_EPS, "bd")


# ---------------------------------------------------------------------------
# Part B: random main body (reproducible per-case), category-weighted.
# ---------------------------------------------------------------------------
ALIGNED_D = [
    8,
    16,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
]
ODD_D = [
    1,
    3,
    7,
    15,
    17,
    31,
    33,
    63,
    65,
    100,
    127,
    129,
    255,
    257,
    333,
    511,
    700,
    1000,
    1023,
    1500,
    2000,
    3000,
    3333,
    5000,
    5001,
    6143,
]
FULL_D = ALIGNED_D + ODD_D
SPLIT_D = [6145, 6400, 7000, 7168, 8000, 8192, 9973, 10000, 12000, 12288, 14000, 16384]
OUTER_FULL = [
    1,
    2,
    3,
    4,
    7,
    8,
    13,
    16,
    31,
    32,
    48,
    64,
    65,
    100,
    128,
    200,
    256,
    512,
    1000,
    2048,
    4096,
]
OUTER_SPLIT = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
OUTER_AXIS = [1, 2, 3, 4, 8, 16]
STRIDED_D = [2, 3, 4, 8, 16, 17, 32, 33, 64, 100, 128, 256]
INNER_AXIS = [2, 3, 4, 5, 7, 8, 9, 16, 32]


def cap_numel(dims, protect_from):
    """Shrink outer axes (indices < protect_from) to 1 until product <= MAX_NUMEL."""
    dims = list(dims)
    i = 0
    while numel(dims) > MAX_NUMEL and i < protect_from:
        dims[i] = 1
        i += 1
    return dims


def draw_full(rng):
    kind = rng.choice(["2d", "2d", "2d", "3d", "1d"])
    d = rng.choice(FULL_D)
    if kind == "1d":
        return (d,), rng.choice([0, -1])
    if kind == "2d":
        outer = rng.choice(OUTER_FULL)
        dims = cap_numel([outer, d], 1)
        return tuple(dims), rng.choice([1, -1])
    # 3d: [a, b, D] dim=last (inner==1)
    a = rng.choice(OUTER_AXIS)
    b = rng.choice(OUTER_FULL)
    dims = cap_numel([a, b, d], 2)
    return tuple(dims), rng.choice([2, -1])


def draw_split(rng):
    d = rng.choice(SPLIT_D)
    if rng.random() < 0.8:
        outer = rng.choice(OUTER_SPLIT)
        dims = cap_numel([outer, d], 1)
        return tuple(dims), rng.choice([1, -1])
    a = rng.choice([1, 2, 3, 4])
    b = rng.choice([1, 2, 3, 4])
    dims = cap_numel([a, b, d], 2)
    return tuple(dims), rng.choice([2, -1])


def draw_strided(rng):
    rank = rng.choice([3, 4, 4, 4, 5])
    dimpos = rng.randint(1, rank - 2)  # middle axis: guarantees >=1 trailing axis
    dims = []
    for ax in range(rank):
        if ax < dimpos:
            dims.append(rng.choice(OUTER_AXIS))
        elif ax == dimpos:
            dims.append(rng.choice(STRIDED_D))
        else:
            dims.append(rng.choice(INNER_AXIS))
    dims = cap_numel(dims, dimpos)
    dim = dimpos if rng.random() < 0.7 else dimpos - rank
    return tuple(dims), dim


def draw_empty(rng):
    rank = rng.choice([1, 2, 2, 3, 3, 4])
    dims = [rng.choice([1, 2, 3, 4, 8, 16]) for _ in range(rank)]
    dims[rng.randint(0, rank - 1)] = 0
    dim = rng.randint(0, rank - 1)
    if rng.random() < 0.3:
        dim -= rank
    return tuple(dims), dim


def build_random(rows, seen, count):
    cats = (["full"] * 46) + (["split"] * 16) + (["strided"] * 32) + (["empty"] * 6)
    for i in range(count):
        rng = rng_for(f"l2ng_rand_{i}")
        dtype = rng.choice(list(DTYPES))
        cat = rng.choice(cats)
        if cat == "full":
            shape, dim = draw_full(rng)
        elif cat == "split":
            shape, dim = draw_split(rng)
        elif cat == "strided":
            shape, dim = draw_strided(rng)
        else:
            shape, dim = draw_empty(rng)
        eps = DEFAULT_EPS if rng.random() < 0.9 else rng.choice(EPS_POOL)
        add_case(rows, seen, dtype, shape, dim, eps, f"rand{i:04d}")


def main():
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "generalization.csv"
        )
    )

    rows = []
    seen = set()
    build_systematic(rows, seen)
    build_random(rows, seen, 900)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(HDR)
        w.writerows(rows)

    # ---- coverage report ----
    from collections import Counter

    key_c, dt_c, rank_c = Counter(), Counter(), Counter()
    for r in rows:
        d = dict(zip(HDR, r))
        shape = eval(d["output_shapes"])[0]
        dim = eval(d["attributes"])["dim"][0]
        dtype = eval(d["input_dtypes"])[0]
        k, _, _, _ = classify(shape, dim)
        key_c[k] += 1
        dt_c[dtype] += 1
        rank_c[len(shape)] += 1
    print(f"generated {len(rows)} cases -> {out}")
    print("  by TilingKey:", dict(sorted(key_c.items())))
    print("  by dtype    :", dict(sorted(dt_c.items())))
    print("  by rank     :", dict(sorted(rank_c.items())))


if __name__ == "__main__":
    main()
