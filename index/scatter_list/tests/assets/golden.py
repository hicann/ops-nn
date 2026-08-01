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

"""TTK custom golden plugin for scatter_list (kernel mode).

Computation formula (docs/aclnnScatterList.md, 计算公式 section):
    var[i][.., indice[i] + k, ..] = updates[i][.., k, ..]   along `axis`
i.e. for each list element i, updates values are scattered into var along the
scatter axis starting at indice[i]. reduce='update' overwrites. An optional mask
(uint8, one entry per var-list element) gates whether each element is updated.

TTK passes each logical INPUT as one positional arg; a TensorList input arrives as
a Python list of ndarrays. Input order = scatter_list_def.cpp Input registration:
    var (TensorList of B tensors), indice, updates, mask(optional)
Output: var (TensorList, in-place).

updates.shape == [B] + var_tensor.shape (with scatter-axis size S2 <= var axis size),
so the scatter axis index on updates = axis, and on each var tensor = axis-relative
to the var rank (updates has one extra leading B dim).

实现说明: 散射写入由 numpy 逐行切片赋值改为 torch 张量赋值（沿 scatter 轴把 updates 的前
n_rows 行写到 var 的 [start, start+n_rows)）——纯 numpy 公式实现与被测 kernel 易犯同类错误，
会掩盖 kernel 精度短板。numpy 仅保留 I/O 与 dtype 转换。语义（mask 门控、1-D/2-D indice 的
start/len、reduce='update' 覆盖写、工作 dtype、返回结构）未变。写入按平坦偏移下发而非
Tensor.index_copy_：后者只接受 [0, axis_size) 的索引，无法表达越界起点下的行为，详见下方注释。
"""

import numpy as np
import torch


def __golden_scatter_list(var_list, indice, updates, mask=None, **kwargs):
    axis = int(kwargs.get("axis", -2))

    var_list = [np.asarray(v) for v in var_list]
    indice = np.asarray(indice)
    updates = np.asarray(updates)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1)

    B = len(var_list)

    # ⚠️ 工作 dtype 必须用**原生 dtype**，不能一律转 float32。
    # ScatterList 是纯拷贝算子（按索引窗口把 updates 覆盖进 var），输出应与输入逐位相同；
    # 而 def 注册了 DT_INT32/DT_INT64——fp32 尾数只有 24 位，装不下 2^30 量级的整数：
    # 实测 int32 = 2^30+100 经 float32 中转变成 2^30+128，**静默丢 28**。
    # 参照本身有损时，int32/int64 这两个 dtype 的精度结论根本没被验证过。
    # bfloat16 numpy 侧没有原生类型，只有它走 fp32 桥接（拷贝无损）。
    def _work(a):
        return np.float32 if a.dtype.name == "bfloat16" else a.dtype

    out = [torch.from_numpy(np.ascontiguousarray(v.astype(_work(v)))) for v in var_list]

    # axis relative to updates dims; corresponding var axis is one less (updates has
    # the extra leading B dim).
    nax = axis if axis >= 0 else updates.ndim + axis
    var_nax = nax - 1
    S2 = updates.shape[nax]

    for i in range(B):
        if mask is not None and mask[i] == 0:
            continue
        # 1-D indice carries only the scatter start offset (kernel copies the full
        # updates scatter dim S2). 2-D indice carries [start, len]; the kernel copies
        # exactly `len` rows from updates (DataCopySmallPad: dim2UpdateLen), so the
        # golden must honor that length (capped at S2 to stay within updates).
        if indice.ndim == 2:
            start = int(indice[i][0])
            n_rows = min(int(indice[i][1]), S2)
        else:
            start = int(indice.reshape(-1)[i])
            n_rows = S2
        if n_rows <= 0:
            continue
        upd_i = torch.from_numpy(
            np.ascontiguousarray(updates[i].astype(_work(var_list[i])))
        )
        # indice 给出的是散射起点，算子按线性地址写入 var：起点乘上 scatter 轴的行跨度，
        # 加上前导维偏移，得到平坦偏移后整行拷贝。README「indice值域：不支持索引越界」，
        # 越界属契约外输入，算子不做边界检查、也没有负索引回绕语义——超出 scatter 轴长时
        # 平坦偏移会落到同一 tensor 的相邻前导维上，整段越出该 tensor 时才写到它之外
        # (此时 var 自身不变)。故这里按平坦偏移赋值：torch 的 index_copy_ 只接受
        # [0, axis_size) 的索引、越界直接抛异常，无法表达上述行为。
        vt = out[i]
        axis_size = vt.shape[var_nax]
        lead = int(np.prod(vt.shape[:var_nax])) if var_nax > 0 else 1
        trail = int(np.prod(vt.shape[var_nax + 1 :])) if var_nax + 1 < vt.ndim else 1
        numel = vt.numel()
        base = (
            torch.arange(lead, dtype=torch.int64).unsqueeze(1) * (axis_size * trail)
            + (start + torch.arange(n_rows, dtype=torch.int64)).unsqueeze(0) * trail
        )  # (lead, n_rows) 每行写入的起始平坦偏移
        valid = (base >= 0) & (base + trail <= numel)
        if not bool(valid.any()):
            continue
        flat_idx = base.unsqueeze(-1) + torch.arange(
            trail, dtype=torch.int64
        )  # (lead,n_rows,trail)
        src = upd_i.reshape(lead, -1, trail)[:, :n_rows, :]
        sel = valid.unsqueeze(-1).expand_as(flat_idx)
        vt.reshape(-1)[flat_idx[sel]] = src[sel]

    var_dtype = var_list[0].dtype
    return [o.numpy().astype(var_dtype) for o in out]


__golden__ = {"kernel": {"scatter_list": "__golden_scatter_list"}}
