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

"""TTK custom golden plugin for scatter_mul (kernel mode).

Compute formula (docs/aclnnScatterMul.md, 计算公式节):
    varRef[indices[i], ...] = varRef[indices[i], ...] * updates[i, ...]
若多个 updates 作用到同一切片，则在该切片上连乘。
索引越界（idx < 0 或 idx >= var.shape[0]）按 kernel 语义跳过。

实现说明: 计算由 numpy 逐条连乘改为 torch 竞品算子 Tensor.index_reduce_(reduce="prod",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误，会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义（重复索引连乘、越界跳过、累加 dtype、返回结构）未变。
"""

import numpy as np
import torch

_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def __golden_scatter_mul(*input_arrays, **kwargs):
    # input order matches CSV input_shapes: var, indices, updates
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtype = var.dtype
    # accumulate in a wider type to match the kernel's internal fp32/int32 precision
    if np.issubdtype(out_dtype, np.floating):
        acc_dtype = np.float32
    else:
        acc_dtype = np.int64

    result = var.astype(acc_dtype).copy()
    upd = updates.astype(acc_dtype)

    var_first = result.shape[0] if result.ndim >= 1 else 1
    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates leading dims correspond to indices entries; trailing = var.shape[1:]
    n_idx = idx_flat.shape[0]
    slice_shape = result.shape[1:]
    upd_slices = (
        upd.reshape((n_idx,) + tuple(slice_shape))
        if n_idx > 0
        else upd.reshape((0,) + tuple(slice_shape))
    )

    # out-of-bound indices dropped first (scatter_reduce_common_simt.h:112), the rest
    # go through the torch reference op; include_self=True == var * all matching updates.
    result_t = torch.from_numpy(result)
    idx_t = torch.from_numpy(idx_flat)
    valid = (idx_t >= 0) & (idx_t < var_first)
    if int(valid.sum()) > 0:
        upd_t = torch.from_numpy(upd_slices)
        result_t.index_reduce_(0, idx_t[valid], upd_t[valid], "prod", include_self=True)

    return [result_t.numpy().astype(out_dtype)]


__golden__ = {"kernel": {"scatter_mul": "__golden_scatter_mul"}}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 保留原 golden，补三方标杆与自定义输入。
# third_party 用 torch 竞品算子在设备侧跑，供 cross_check 比对；
# customize_inputs 即原 input.py 的合法索引重采样（原文件保留，不影响旧机制）。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。

    ⚠️ 不能把 fp16/bf16 升 fp32 再算。本算子对重复索引是**链式**运算, 误差随链长累积;
    三方若用更高精度跑, cross_check 拿到的就是"内核误差 / 一个 fp16 实现物理上够不到的
    参照"之比, 会系统性判红——实测 scatter_div fp16 链长中位 8 的用例 mare_ratio 12.78
    (限值 5), 改回与算子同 dtype 的语义后 0.65 通过, 内核本身的误差 99.99% 落在 fp16
    链式误差预算内。三方必须与算子同 dtype 语义, 内核误差才有可比对象。
    golden 反过来要保持高精度(fp32 链), 它是两条腿共同的参照点。
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    t = torch.as_tensor(np.asarray(x))  # 仅本地自测兜底: 框架侧不会走到
    return t.to(torch.float32) if t.dtype == torch.bfloat16 else t.clone()


def scatter_mul_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_mul.
    All the parameters (names and order) follow scatter_mul_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Default random indices may fall out of [0, var.shape[0]); resample them into
    the legal first-dim range so var[indices[i]] is always addressable (kernel
    silently skips out-of-range indices, but golden/kernel agree only on legal ones).

    Args:
        **kwargs: input_dtypes, full_soc_version, short_soc_version, testcase_name

    Returns:
        Input tensors
    """
    shape_indices, dtype_indices, size_indices = (
        indices.shape,
        indices.dtype,
        indices.size,
    )
    max_indices = var.shape[0]

    if var.size * indices.size * updates.size == 0:
        return [var, indices, updates]

    replace = size_indices > max_indices
    indices = np.random.choice(max_indices, size_indices, replace=replace).astype(
        dtype_indices
    )
    indices = np.reshape(indices, shape_indices)
    return [var, indices, updates]


class _ScatterMulCompose:
    def __call__(self, var, indices, updates, **kwargs):
        work = _tp_t(var)
        upd = _tp_t(updates).reshape((-1,) + tuple(work.shape[1:]))
        it = (
            indices
            if isinstance(indices, torch.Tensor)
            else torch.as_tensor(np.asarray(indices))
        )
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx = idx[valid]
        if idx.numel() > 0:
            work = work.index_reduce(0, idx, upd[valid], "prod", include_self=True)
        return [work]


_GOLDEN_FN = __golden_scatter_mul


class ScatterMulKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterMulCompose}
    customize_inputs = scatter_mul_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_mul": "ScatterMulKernelSpec",
    "aclnnScatterMul": "ScatterMulAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路: 框架传入的已是设备侧 torch.Tensor, 不经 numpy。"""
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


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


class ScatterMulAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterMulGetWorkspaceSize 的
    形参 varRef/indices/updates/useLocking;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 用适配类把头文件的
    varRef 接到 kernel 通路竞品类的 def 注册名 var 上。"""

    @staticmethod
    def golden(varRef, indices, updates, useLocking=None, **kwargs):
        work = _tp_one(varRef).clone()
        upd = _tp_one(updates).reshape((-1,) + tuple(work.shape[1:]))
        it = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices)
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx = idx[valid]
        if idx.numel() > 0:
            work = work.index_reduce(0, idx, upd[valid], "prod", include_self=True)
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMulCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin, 但 TF 不是 TestSpec 的注册通路
# (README 四通路为 Kernel/GEIR/ACLNN/E2E), 如需 TF 对标应以 third_party 的 tf
# vendor 形式补充, 本次未做。
# e2e / ONNX / 融合 pass: 均未交付。
