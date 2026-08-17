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
# 生成 MultilabelMarginLoss 全泛化用例 CSV（覆盖 dtype×reduction×1D/2D×密度×对齐×边界）。
# 输出/输入 shape 规则严格对齐 infershape：none+2D->[N], none+1D 或 mean/sum -> 标量[1]；is_target=x.shape。
import random

random.seed(20260719)

DTYPES = {
    "float32": (0.0001, 0.0001),
    "float16": (0.001, 0.001),
    "bfloat16": (0.004, 0.004),
}
REDUCTIONS = ["none", "mean", "sum"]
HDR = (
    "testcase_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,"
    "output_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision"
)


def y_shape(red, ndim, n):
    return f"({n},)" if (red == "none" and ndim == 2) else "(1,)"


def row(name, dt, red, ndim, n, c):
    tol = DTYPES[dt]
    if ndim == 2:
        xs = f"({n}, {c})"
    else:
        xs = f"({c},)"
    ish = f'"({xs},{xs})"'
    idt = f"\"('{dt}','int32')\""
    fmt = "\"('ND','ND')\""
    osh = f'"({y_shape(red, ndim, n)},{xs})"'
    odt = f"\"('{dt}','int32')\""
    attr = f"\"{{'reduction': '{red}'}}\""
    rng = f'"((-2, 2),(-1, {c}))"'
    ptol = f'"(({tol[0]}, {tol[1]}),(0, 0))"'
    return f"{name},multilabel_margin_loss,{ish},{idt},{fmt},{osh},{odt},{fmt},{attr},{rng},{ptol},0"


lines = [HDR]
idx = 0

# --- 随机主体 ~900：dtype×reduction×1D/2D，N/C 覆盖单核~多核、对齐/非对齐 ---
for _ in range(900):
    dt = random.choice(list(DTYPES))
    red = random.choice(REDUCTIONS)
    ndim = random.choice([1, 2, 2, 2])  # 2D 权重更高
    c = random.choice(
        [
            1,
            2,
            3,
            7,
            8,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            100,
            127,
            128,
            200,
            255,
            256,
            257,
            384,
            512,
        ]
    )
    n = (
        1
        if ndim == 1
        else random.choice([1, 2, 3, 4, 8, 16, 32, 48, 56, 64, 65, 96, 128])
    )
    idx += 1
    lines.append(
        row(f"mll_rand_{idx:04d}_{dt[:3]}_{red}_{ndim}d_n{n}c{c}", dt, red, ndim, n, c)
    )

# --- 边界/特例 ~100：极小 C、单类、大多核、对齐成对、各 dtype×reduction 网格 ---
BOUNDARY = []
for dt in DTYPES:
    for red in REDUCTIONS:
        BOUNDARY += [
            (dt, red, 2, 1, 1),  # 最小 2D
            (dt, red, 1, 1, 1),  # 最小 1D 单元素
            (dt, red, 2, 1, 8),  # 单行
            (dt, red, 2, 64, 256),  # 多核 blockdim 大
            (dt, red, 2, 128, 512),  # 更大多核+大 C
            (dt, red, 2, 4, 17),  # 非对齐 C
            (dt, red, 2, 4, 16),  # 对齐 C 成对
            (dt, red, 1, 1, 1024),  # 1D 长向量
            (dt, red, 2, 33, 65),  # 非对齐 N+C
        ]
for dt, red, ndim, n, c in BOUNDARY:
    idx += 1
    lines.append(
        row(f"mll_bd_{idx:04d}_{dt[:3]}_{red}_{ndim}d_n{n}c{c}", dt, red, ndim, n, c)
    )

# --- 规模/边界补齐 ~200：覆盖原随机块封顶(N<=128, C<=512)之外的区间 ---
# 补这一段的原因:原用例集 N 封顶 128、C 封顶 1024,而 arch35 内核按 host 下发的 ubFactor
# 对逐行损失做分块暂存与回读 —— ubFactor = min(UB余量/单元素开销, 对齐后的 N),
# 只有 N 超过 UB 余量算出的上限(小 C 时约 1.5 万)才会真正走多块路径。
# 原集合一条都进不了该分支,等于分块逻辑无覆盖;同理 C 越大 ubFactor 越小,
# 大 C 也要覆盖以检验 host UB 预算既不误判失败、也不低估导致越界。
# 追加在随机块之后,不改动上面任何取值池 —— 保证既有 981 例逐例可复现。
SCALE = []
for dt in DTYPES:
    for red in REDUCTIONS:
        SCALE += [
            # 大 N:跨越 ubFactor 上限,逼出多块分支(none 每行一个输出,回读压力最大)
            (dt, red, 2, 4096, 8),
            (dt, red, 2, 16384, 8),
            (dt, red, 2, 28672, 16),
            (dt, red, 2, 20000, 33),  # 大 N + 非对齐 C
            # 大 C:ubFactor 随 C 变小,同时检验 host UB 预算边界
            (dt, red, 2, 2, 1024),
            (dt, red, 2, 2, 2048),
            (dt, red, 2, 4, 4096),
            (dt, red, 1, 1, 4096),  # 1D 超长向量
            # 多核切分边界:恰好等于/相邻于核数,pivot 分支两侧
            (dt, red, 2, 63, 64),
            (dt, red, 2, 64, 64),
            (dt, red, 2, 65, 64),
            (dt, red, 2, 129, 32),
            # 大 N + 大 C 组合(UB 与分核同时吃紧)
            (dt, red, 2, 4096, 512),
        ]
# 注:不纳入 N=0(空 batch)。实测 torch.ops.aten.multilabel_margin_loss_forward 在空输入下,
# reduction=mean 返回未初始化值(如 -1.9818e+08,非 nan 非 0),不构成可用真值;none/sum 虽良定义,
# 但空输出的逐元素比对在 TTK 里记为 0.0%。空 batch 属需先定契约的 DFX 边界项,不是精度泛化能覆盖的。
# 内核侧已有防御性处理:tiling 令 usedCoreNum=1 维持合法 grid,kernel 零行不写。
for dt, red, ndim, n, c in SCALE:
    idx += 1
    lines.append(
        row(f"mll_sc_{idx:04d}_{dt[:3]}_{red}_{ndim}d_n{n}c{c}", dt, red, ndim, n, c)
    )

with open("mll_generalization.csv", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"generated {len(lines) - 1} cases -> mll_generalization.csv")
