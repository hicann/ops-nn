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
#   7000  FullLoad  : inner == 1 and AlignUp(D, VL) <= derive_ub_factor()   (ascend950: D <= 6080)
#   7010  SplitD    : inner == 1 and AlignUp(D, VL)  > derive_ub_factor()
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
    "batch_seed",
]


BATCH_SEED = "20260902"  # 固定输入数据种子:无此列则 TTK 每次生成不同输入,结果不可复现
UB_SIZE = 253952  # Ascend950 ub_size,取自 platform_config/Ascend950DT_950x.ini
# (曾误写 245760 = 已减过一次 UB_RESERVED,导致 7000/7010 选路预测偏一档)
VL_ELEMS = 64  # GetVRegSize()/sizeof(float),与内核 V_LENGTH 一致
FLOAT_BYTE = 4
UB_RESERVED = 8 * 1024
STRIDED_BUF_NUM = 4
BUFFER_NUM = 2
VL_SLACK = 64
QUEUE_AND_MID_UNITS = (
    STRIDED_BUF_NUM * BUFFER_NUM + 2
)  # 8 队列缓冲 + 2 块 fp32 中间量 tile
MAX_BLOCK_COUNT = 65535  # DataCopyExtParams::blockCount 为 uint16


def rng_for(key):
    """Deterministic per-case RNG seeded from md5(key) (reproducible, order-independent)."""
    import random

    seed = int(hashlib.md5(str(key).encode()).hexdigest(), 16) & 0xFFFFFFFF
    return random.Random(seed)


def resolve_dim(dim, rank):
    d = dim + rank if dim < 0 else dim
    return d


def resolve_axes(dim, rank):
    """折负 + 去重 + 排序,返回轴元组;空元组表示不归约(对齐 host ResolveDimAndShape)。"""
    vals = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    return tuple(sorted({(v + rank if v < 0 else v) for v in vals}))


def classify(shape, dim, dtype="float32"):
    """Replicates op_host SelectTemplate: returns (tilingkey, outer, D, inner).

    dim 可为标量或列表;列表按折负/去重/排序后须构成连续区间(非连续在 host 被拒收,不进本集)。
    空集 = 不归约 -> outer=totalNum, D=1, inner=1。
    """
    rank = len(shape)
    axes = resolve_axes(dim, rank)
    total = 1
    for s in shape:
        total *= s
    if total == 0:
        return 8000, 0, 0, 0
    if not axes:
        return 7000, total, 1, 1
    lo, hi = axes[0], axes[-1]
    assert hi - lo + 1 == len(axes), f"non-contiguous dim {dim} on rank {rank}"
    outer = 1
    for i in range(lo):
        outer *= shape[i]
    dlen = 1
    for i in range(lo, hi + 1):
        dlen *= shape[i]
    inner = 1
    for i in range(hi + 1, rank):
        inner *= shape[i]
    if inner == 1:
        row_align_vl = -(-dlen // VL_ELEMS) * VL_ELEMS
        key = 7000 if row_align_vl <= derive_ub_factor() else 7010
    else:
        # 7020 整段 D 常驻 UB;装不下则 7030 沿 D 分块。
        # 与 host SelectStridedTemplate **逐项同式**:分母是 QUEUE_AND_MID_UNITS(=10)
        # —— 4 队列 x 双缓冲(8) + 2 块 fp32 中间量 tile,不是 8。少算这 2 块会把预算高估 25%,
        # 导致本该 7030 的 shape 被标成 7020(实测 (1,920,8)/(1,921,8) 两例即此)。
        block = 8 if dtype == "float32" else 16
        u_elems = (UB_SIZE - UB_RESERVED) // FLOAT_BYTE
        col_align_cap = (
            ((u_elems - 12 * VL_ELEMS) // (QUEUE_AND_MID_UNITS * dlen + 2))
            if dlen > 0
            else inner
        )
        max_col = (col_align_cap // block) * block
        key = 7020 if (max_col >= block and dlen <= MAX_BLOCK_COUNT) else 7030
    return key, outer, dlen, inner


def derive_ub_factor():
    """full_load 单次可处理元素数 —— 与 host DeriveUbFactor(ubSize, isSplitD=False) 同式。

    占用 = 4 队列 x 双缓冲 x 4B + 2 x reduceBuf(<=F x 4B) = 40B/元素,另扣两个 tmp buf。
    阈值必须由 ubSize 解出而非写死:写死一个数就会像 6144 那样与实现悄悄偏离,
    用例名标 k7000 实际却跑 7010,边界档从此零覆盖且没人发现。
    """
    ub_avail = UB_SIZE - UB_RESERVED
    per_elem = STRIDED_BUF_NUM * BUFFER_NUM * FLOAT_BYTE + 2 * FLOAT_BYTE
    fixed = 2 * VL_ELEMS * FLOAT_BYTE
    f = (ub_avail - fixed) // per_elem
    f = (f // VL_ELEMS) * VL_ELEMS
    return f if f >= VL_ELEMS else VL_ELEMS


# full_load 能吃下的最大归约轴长度(边界档一律以它为基准表达,避免再次写死漂移)
_FULL_MAX = derive_ub_factor()


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _build_attrs(dim, eps):
    """attributes 字典;值为 None 表示**该属性不下发** —— 走 proto 默认值那条缺省路径。

    OPTIONAL 属性的缺省档必须真的不传:每例都显式传值时,proto 默认值(dim={} / eps=1e-4)
    这条最常走的路径反而零覆盖。
    """
    attrs = {}
    if dim is not None:
        attrs["dim"] = [
            int(v) for v in (dim if isinstance(dim, (list, tuple)) else [dim])
        ]
    if eps is not None:
        attrs["eps"] = eps
    return attrs


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
        "attributes": repr(_build_attrs(dim, eps)),
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
        # TTK 的 --seed 只在用例集有 batch_seed 列时才生效;没有这列输入每次都不同、跑批不可复现
        "batch_seed": BATCH_SEED,
    }
    return [row[c] for c in HDR]


def eps_str(eps):
    return "e" + repr(eps).replace("-", "m").replace(".", "p").replace("+", "")


def add_case(rows, seen, dtype, shape, dim, eps, prefix):
    # dim 缺省 == 传空数组 == 不归约(GE 通路语义,见 01 §6.2),分类按空集走
    key, outer, dlen, inner = classify(shape, [] if dim is None else dim, dtype)
    shp_s = "x".join(str(s) for s in shape)
    dim_s = (
        "na"
        if dim is None
        else (
            ("n" if not dim else "_".join(str(v) for v in dim))
            if isinstance(dim, (list, tuple))
            else str(dim)
        ).replace("-", "m")
    )
    eps_s = "_epsna" if eps is None else ""
    base = f"{OP_NAME}_{prefix}_{TAG[dtype]}_k{key}_{shp_s}_d{dim_s}{eps_s}"
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
        (13, _FULL_MAX, 1),  # D == threshold, still 7000
        (16, _FULL_MAX - 1, 1),  # just under threshold, non-aligned
        (2048, 32, 1),  # many rows, tiny D (deep multi-core)
        (33, 4096, -1),  # aligned large D
        (128, 257, 1),  # non-aligned prime-ish D
    ]
    for dt in dts:
        for outer, d, dim in full_specs:
            add_case(rows, seen, dt, (outer, d), dim, DEFAULT_EPS, "bd")

    # A2  7010 split_d (inner==1, 整行装不下): threshold+1, pow2/non-pow2, probe-the-ceiling large D.
    split_specs = [
        (1, _FULL_MAX + 1, 1),  # just over threshold, single group/core
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
    _FULL_MAX,
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
    _FULL_MAX - 1,
]
FULL_D = ALIGNED_D + ODD_D
SPLIT_D = [
    _FULL_MAX + 1,
    6400,
    7000,
    7168,
    8000,
    8192,
    9973,
    10000,
    12000,
    12288,
    14000,
    16384,
]
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


def mutate_dim(rng, shape, dim):
    """把随机档抽到的单轴 dim 按概率变形成:连续多轴 / 空集 / 冗余写法(重复·乱序·负值)。

    连续多轴以原轴为锚向两侧扩展,保证折算去重后仍是连续区间(非连续是拒收档,不进泛化集)。
    """
    rank = len(shape)
    if rank == 0:
        return dim
    r = rng.random()
    if r < 0.05:
        return []  # 空集:不归约
    d = dim + rank if dim < 0 else dim
    if r < 0.25 and rank >= 2:
        lo = hi = d
        span = rng.choice([2, 2, 3, rank])
        while hi - lo + 1 < span:
            if lo > 0 and (hi == rank - 1 or rng.random() < 0.5):
                lo -= 1
            elif hi < rank - 1:
                hi += 1
            else:
                break
        axes = list(range(lo, hi + 1))
        if rng.random() < 0.3:
            rng.shuffle(axes)  # 乱序传入,折算后应等价
        if rng.random() < 0.2:
            axes = [a - rank for a in axes]  # 全负值写法
        return axes
    if r < 0.30:
        k = rng.choice([2, 3, 20])
        return [dim] * k  # 冗余重复,去重后等价单轴
    return dim


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
        # dim 形态与 shape/rank/dtype 交叉抽样:单轴 70% / 连续多轴 20% / 空集 5% / 冗余写法 5%。
        # (只在确定性档手写几组多轴 = 新支持面没进随机交叉,正是 R6 画像式交叉要挡的形态)
        dim = mutate_dim(rng, shape, dim)
        eps = DEFAULT_EPS if rng.random() < 0.9 else rng.choice(EPS_POOL)
        add_case(rows, seen, dtype, shape, dim, eps, f"rand{i:04d}")


# ---------------------------------------------------------------------------
# Part C: dim 语义档(本轮重构新增支持面)
#   R2 缺省档   —— 不传/传空 dim = 不归约,×rank 1..8
#   连续多轴档  —— [lo..hi] 各种长度与位置,含全轴规约
#   等价档      —— 重复/乱序/正负混写/长度上限,须与规范形式同结果
#   7030 档     —— inner>1 且整段 D 放不下 UB(老集在这个交叉上零覆盖)
# 非连续轴集是拒收档,不进泛化集(由 tiling UT 覆盖)。
# ---------------------------------------------------------------------------
def build_dim_semantics(rows, seen):
    dts = list(DTYPES)

    # C1 空集(不归约) × rank 1..8 × dtype
    empty_shapes = [
        (1024,),
        (32, 512),
        (4, 8, 16),
        (2, 4, 8, 16),
        (2, 2, 4, 8, 16),
        (2, 2, 2, 4, 8, 8),
        (2, 2, 2, 2, 4, 4, 8),
        (2, 2, 2, 2, 2, 2, 4, 4),
    ]
    for dt in dts:
        for shp in empty_shapes:
            add_case(rows, seen, dt, shp, [], DEFAULT_EPS, "c1empty")
    # C1b 空集 + 非对齐 / 单元素 / 空张量
    for dt in dts:
        for shp in [(1,), (33,), (7, 13), (3, 0, 5)]:
            add_case(rows, seen, dt, shp, [], DEFAULT_EPS, "c1empty")

    # C1c **属性缺省档(R2)**:dim / eps 真的不下发,走 proto 默认值那条路径。
    #     不传 dim ≡ 传空数组 ≡ 不归约(GE 通路语义);不传 eps ≡ 1e-4。
    #     此前 1184 例全部显式带 dim 与 eps,这条最常走的路径零覆盖(code-check G26 命中)。
    default_shapes = [
        (1024,),
        (32, 512),
        (4, 8, 16),
        (2, 4, 8, 16),
        (33,),
        (7, 13),
        (3, 0, 5),
    ]
    for dt in dts:
        for shp in default_shapes:
            add_case(rows, seen, dt, shp, None, DEFAULT_EPS, "c1def")  # dim 缺省
            add_case(rows, seen, dt, shp, None, None, "c1def")  # dim + eps 均缺省
        # eps 缺省但 dim 显式:分离两个属性的缺省路径
        for shp, dim in [((64, 256), 1), ((8, 16, 32), -1), ((4, 8, 16, 2), 2)]:
            add_case(rows, seen, dt, shp, dim, None, "c1def")

    # C2 连续多轴:位置(头/中/尾) × 长度(2/3/全) × inner==1 与 inner>1
    multi_specs = [
        ((4, 8, 16), [0, 1]),
        ((4, 8, 16), [1, 2]),
        ((4, 8, 16), [0, 1, 2]),
        ((2, 4, 8, 16), [0, 1]),
        ((2, 4, 8, 16), [1, 2]),
        ((2, 4, 8, 16), [2, 3]),
        ((2, 4, 8, 16), [0, 1, 2]),
        ((2, 4, 8, 16), [1, 2, 3]),
        ((2, 4, 8, 16), [0, 1, 2, 3]),
        ((16, 16), [0, 1]),  # A2 自己 ST 的同款
        ((3, 5, 7), [0, 1]),
        ((3, 5, 7), [1, 2]),  # 全非对齐
        ((2, 2, 3, 65), [1, 2]),  # 折叠后 D=6 尾轴大
        ((2, 65, 3, 2), [1, 2]),  # 折叠后 D=195, inner=2
    ]
    for dt in dts:
        for shp, dim in multi_specs:
            add_case(rows, seen, dt, shp, dim, DEFAULT_EPS, "c2multi")

    # C3 等价档:与规范形式同结果(重复/乱序/负值/正负混写/长度上限 20)
    equiv_specs = [
        ((4, 8, 16), [1]),
        ((4, 8, 16), [1, 1, 1]),
        ((4, 8, 16), [-2]),
        ((4, 8, 16), [1, -2]),
        ((4, 8, 16), [1] * 20),
        ((4, 8, 16), [2, 1]),
        ((4, 8, 16), [-1, -2]),
        ((2, 4, 8, 16), [3, 2]),
        ((2, 4, 8, 16), [-1, -2, -3]),
    ]
    for dt in dts:
        for shp, dim in equiv_specs:
            add_case(rows, seen, dt, shp, dim, DEFAULT_EPS, "c3equiv")

    # C4 7030 档:inner>1 且 D 大(整段放不下 UB) —— 老集零覆盖的交叉
    split_specs = [
        ((2, 40000, 2), 1),
        ((1, 70000, 2), 1),
        ((2, 20000, 4), 1),
        ((1, 1000, 2), 1),
        ((1, 921, 8), 1),
        ((1, 920, 8), 1),  # 阈值两侧
        ((3, 5000, 3), 1),
        ((1, 100000, 2), 1),
        ((2, 3, 8000, 2), 2),
        ((2, 8000, 3, 2), 1),
    ]
    for dt in dts:
        for shp, dim in split_specs:
            add_case(rows, seen, dt, shp, dim, DEFAULT_EPS, "c4splitd")
    # C6 D(归约轴长) x groups(归约组数) 的四象限补齐
    #   1184 例老集里 D>=1000 且 groups>=3000 这一格恰好是 0 例(被单例元素数封顶挤掉),
    #   而 issue #31 的 AI 分析恰好点在这片区域(作者已否定它是触发条件,但格子不能空)。
    #   每例 <= 1.7e7 元素(fp32 约 67MB/张量),显存可承受。
    for dt in dts:
        for shp, dim, rm in [
            ((3000, 1000), 1, "D=1000 groups=3000 inner==1"),
            ((4096, 4096), 1, "D=4096 groups=4096 inner==1"),
            ((10000, 1000), 1, "D=1000 groups=1e4 inner==1"),
            ((3000, 2000), 1, "D=2000 groups=3000 inner==1"),
            ((2000, 1000, 2), 1, "D=1000 groups=4000 inner=2 -> strided 族"),
            ((1000, 2000, 4), 1, "D=2000 groups=4000 inner=4 -> strided 族"),
            ((3000, 1000), 0, "D=3000 groups=1000 头部归约"),
            ((50, 1000, 100), 1, "D=1000 groups=5000 inner=100"),
        ]:
            add_case(rows, seen, dt, shp, dim, DEFAULT_EPS, "c6quad")
    # C6b 折叠后落进该象限
    for dt in dts:
        add_case(rows, seen, dt, (2000, 50, 40), [1, 2], DEFAULT_EPS, "c6quad")
        add_case(rows, seen, dt, (100, 40, 50, 20), [1, 2], DEFAULT_EPS, "c6quad")

    # C5 空张量逐轴枚举:rank 1..8 × 每个轴位置各置 0 一次(确定性,补随机档的轴位空洞)
    for dt in dts:
        for rank in range(1, 9):
            base = [2] * rank
            for ax in range(rank):
                shp = list(base)
                shp[ax] = 0
                add_case(
                    rows,
                    seen,
                    dt,
                    tuple(shp),
                    min(1, rank - 1),
                    DEFAULT_EPS,
                    "c5empty0",
                )

    # C4b 折叠后才变大的 D(连续多轴 → 7030)
    for dt in dts:
        for shp, dim in [((2, 200, 300, 2), [1, 2]), ((1, 100, 1000, 4), [1, 2])]:
            add_case(rows, seen, dt, shp, dim, DEFAULT_EPS, "c4splitd")


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
    build_dim_semantics(rows, seen)

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
        dim = eval(d["attributes"]).get("dim", [])  # 缺省档没有 dim 键,等价空集
        dtype = eval(d["input_dtypes"])[0]
        k, _, _, _ = classify(shape, dim, dtype)
        key_c[k] += 1
        dt_c[dtype] += 1
        rank_c[len(shape)] += 1
    print(f"generated {len(rows)} cases -> {out}")
    print("  by TilingKey:", dict(sorted(key_c.items())))
    print("  by dtype    :", dict(sorted(dt_c.items())))
    print("  by rank     :", dict(sorted(rank_c.items())))


if __name__ == "__main__":
    main()
