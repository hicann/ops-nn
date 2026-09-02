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

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）。
# third_party 用 torch 原生的 narrow().copy_() 表达"按索引窗口整段覆盖"，与上面 golden
# 手写的平坦偏移算术（算 base/flat_idx 再展平赋值）是两条独立实现，比对有意义。
# 覆盖范围限于契约内的合法索引：README「indice值域：不支持索引越界」，越界属契约外，
# torch 的切片/index_copy_ 也表达不了算子那种"越界落到相邻前导维"的平坦写行为。
# 判据保持 binary_equal：纯拷贝语义就该逐位相同，cross_check 反而是降级。
# customize_inputs 即原 input.py 的合法索引重采样（原文件保留，不影响旧机制）。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


def scatter_list_input(var_list, indice, updates, *rest, axis=-2, **kwargs):
    """
    Input function for scatter_list.
    All the parameters (names and order) follow scatter_list_def.cpp without outputs.
    var is a TensorList (list of ndarrays); indice/updates/mask are ndarrays.

    Resample `indice` so every scatter window [start, start + S2) stays inside the
    var scatter axis. 1-D indice carries only the start offset; 2-D indice carries
    [start, len] -> keep len, clamp start so start + len <= var_axis_size.

    Returns:
        Input tensors (var_list, indice, updates[, mask]) —— 槽位数与入参一致
    """
    var0 = np.asarray(var_list[0])
    updates = np.asarray(updates)
    nax = axis if axis >= 0 else updates.ndim + axis
    var_nax = nax - 1  # updates has extra leading list dim
    var_axis = var0.shape[var_nax]
    S2 = updates.shape[nax]
    B = len(var_list)

    indice = np.asarray(indice)
    dt = indice.dtype
    if indice.ndim == 2:
        lens = np.minimum(indice[:, 1], S2)
        lens = np.clip(lens, 1, max(1, var_axis))
        highs = np.maximum(var_axis - lens, 0) + 1
        starts = np.array([np.random.randint(0, h) for h in highs], dtype=dt)
        new = np.stack([starts, lens.astype(dt)], axis=1)
    else:
        high = max(var_axis - S2, 0) + 1
        new = np.random.randint(0, high, size=(B,)).astype(dt)
        new = np.reshape(new, indice.shape)
    # 原样回传收到的尾部槽位(mask)。TTK 按 flat_input_shapes 计数校验返回个数,
    # 用例声明了 mask 槽但值为 None 时若把它丢掉, 会 INPUT_GEN_FAILURE:
    # "Input plugin returned 3 arrays, expected 4"(实测 4 例)。
    return [var_list, new, updates, *rest]


class _ScatterListCompose:
    """参数名按 scatter_list_def.cpp 的注册名: var / indice / updates / mask。
    mask 是可选输入、axis 是可选属性, 一律从 kwargs 取（bind_by_name 不认默认值）。"""

    def __call__(self, var, indice, updates, **kwargs):
        mask = kwargs.get("mask")
        axis = int(kwargs.get("axis", -2))

        def _to_t(a):
            t = (
                a
                if isinstance(a, torch.Tensor)
                else torch.as_tensor(np.ascontiguousarray(a))  # 仅本地自测兜底
            )
            return t.to(torch.float32) if t.dtype == torch.bfloat16 else t

        var_list = var if isinstance(var, (list, tuple)) else [var]
        outs = [_to_t(v).clone() for v in var_list]
        upd = _to_t(updates)
        idx = (
            indice
            if isinstance(indice, torch.Tensor)
            else torch.as_tensor(np.asarray(indice))
        )
        msk = None
        if mask is not None:
            mt = (
                mask
                if isinstance(mask, torch.Tensor)
                else torch.as_tensor(np.asarray(mask))
            )
            msk = mt.reshape(-1)

        nax = axis if axis >= 0 else upd.dim() + axis
        var_nax = nax - 1  # updates 比 var 多一个前导 batch 维
        s2 = int(upd.shape[nax])

        for i in range(len(outs)):
            if msk is not None and msk[i] == 0:
                continue
            if idx.dim() == 2:
                start, n_rows = int(idx[i][0]), min(int(idx[i][1]), s2)
            else:
                start, n_rows = int(idx.reshape(-1)[i]), s2
            vt = outs[i]
            if n_rows <= 0 or start < 0 or start + n_rows > vt.shape[var_nax]:
                continue  # 契约外(越界)不比对, 见类注释
            vt.narrow(var_nax, start, n_rows).copy_(upd[i].narrow(var_nax, 0, n_rows))
        return outs


_GOLDEN_FN = __golden_scatter_list


class ScatterListKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterListCompose}
    customize_inputs = scatter_list_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_list": "ScatterListKernelSpec",
    "aclnnScatterList": "ScatterListAclnnSpec",
}


def _keep_dtype(res, ref):
    """golden 输出 dtype 必须与算子输出一致: 比对按 dtype 判定, fp16/bf16 提到 fp32
    算完必须还原, 否则 binary_equal 直接判 "dtype 不可比"(实测 GOLD 0%)。
    golden_mode=Promote 时入参本身已是 fp32, 此处是恒等操作。"""
    refs = ref if isinstance(ref, (list, tuple)) else [ref] * len(res)
    return [
        t.to(r.dtype)
        if isinstance(t, torch.Tensor) and isinstance(r, torch.Tensor)
        else t
        for t, r in zip(res, refs)
    ]


class ScatterListAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterListGetWorkspaceSize 的
    形参 varRef/indice/updates/maskOptional/reduceOptional/axis;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 用适配类把头文件的
    varRef/maskOptional 接到 kernel 通路竞品类的 def 注册名 var/mask 上。
    reduceOptional 只有 "update"(按索引窗口覆盖写)一种语义, 不分支。"""

    @staticmethod
    def golden(
        varRef,
        indice,
        updates,
        maskOptional=None,
        reduceOptional=None,
        axis=-2,
        **kwargs,
    ):
        return _keep_dtype(
            _ScatterListCompose()(
                varRef, indice, updates, mask=maskOptional, axis=axis
            ),
            varRef,
        )

    class _Compose:
        def __call__(self, varRef, indice, updates, **kwargs):
            kw = dict(kwargs)
            kw.setdefault("mask", kw.get("maskOptional"))
            return _ScatterListCompose()(varRef, indice, updates, **kw)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——无 framework/ 插件、无 graph pass,
# 也未发现 torch_npu 绑定到 aclnnScatterList。
