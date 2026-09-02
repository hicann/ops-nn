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

"""TTK custom golden for scatter_min (kernel mode).

计算公式 (依据 aclnnScatterMin.md 「功能说明」节):
    varRef[indices[i], ...] = min(varRef[indices[i], ...], updates[i, ...])
  - indices 为扁平的索引条目序列; 对第 i 个索引条目, 取 var 的第 indices[i] 行
    (slice, 即 var[indices[i]]) 与 updates 的第 i 个 slice 逐元素取 min.
  - 越界索引被跳过 (scatter_reduce_common_simt.h:112: idxVal<0 或 >=var.shape[0] -> skip),
    与 kernel 语义一致.
  - 重复索引依次取 min (顺序无关, min 满足交换律/结合律).
  - 输出为原地更新后的 var (单输出).

输入顺序 (scatter_min_def.cpp): var, indices, updates
输出顺序 (scatter_min_def.cpp / infershape.cpp): var (inplace)

实现说明: 计算由 numpy 逐条 np.minimum 改为 torch 竞品算子 Tensor.index_reduce_(reduce="amin",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误, 会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义(重复索引取最小、越界跳过、工作 dtype、返回结构) 未变。
"""

import numpy as np
import torch


def __golden_scatter_min(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtype = var.dtype
    var_shape = var.shape

    # var.shape[0] = 索引上界; slice = var.shape[1:]
    var_first_dim = var_shape[0] if var_shape else 0
    slice_shape = tuple(var_shape[1:])
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # 空切片 (var.shape 含 0 维 -> slice_size==0): 每个 slice 0 宽, scatter 是 no-op,
    # var 原样返回 (kernel 同样在 sliceSize==0 时直接 return)。不加这道会让下面
    # upd_flat=(0,1) 在循环里 upd_flat[i] 越界 -> IndexError(golden 侧 GOLDEN_FAILURE)。
    if slice_size == 0:
        return [var.astype(out_dtype)]

    # 中间计算用 float32, 整型保持原类型精确比对
    is_float = np.issubdtype(out_dtype, np.floating)
    work_dtype = np.float32 if is_float else out_dtype

    result = (
        var.astype(work_dtype).reshape(var_first_dim, slice_size)
        if var_first_dim
        else var.astype(work_dtype).reshape(0, slice_size)
    )

    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates 扁平化为 (indices_num, slice_size)
    upd_flat = (
        updates.astype(work_dtype).reshape(-1, slice_size)
        if slice_size
        else updates.astype(work_dtype).reshape(-1, 1)
    )

    n = idx_flat.shape[0]
    # 越界索引先剔除 (与 kernel 的 skip 一致), 剩下的交给 torch 竞品算子;
    # include_self=True 即 min(var 原值, 命中该行的所有 updates), 重复索引顺序无关。
    result_t = torch.from_numpy(result)
    idx_t = torch.from_numpy(idx_flat)
    valid = (idx_t >= 0) & (idx_t < var_first_dim)
    if int(valid.sum()) > 0:
        upd_t = torch.from_numpy(upd_flat[:n])
        result_t.index_reduce_(0, idx_t[valid], upd_t[valid], "amin", include_self=True)

    out = result_t.numpy().reshape(var_shape).astype(out_dtype)
    return [out]


__golden__ = {"kernel": {"scatter_min": "__golden_scatter_min"}}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 保留原 golden，补三方标杆与自定义输入。
# third_party 用 torch 竞品算子在设备侧跑，供 cross_check 比对；
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


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    if t.dtype in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    return t.clone()


def scatter_min_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_min.
    All the parameters (names and order) follow scatter_min_def.cpp without outputs.
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


class _ScatterMinCompose:
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
            work = work.index_reduce(0, idx, upd[valid], "amin", include_self=True)
        return [work]


_GOLDEN_FN = __golden_scatter_min


class ScatterMinKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterMinCompose}
    customize_inputs = scatter_min_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_min": "ScatterMinKernelSpec",
    "aclnnScatterMin": "ScatterMinAclnnSpec",
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


class ScatterMinAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterMinGetWorkspaceSize 的
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
            work = work.index_reduce(0, idx, upd[valid], "amin", include_self=True)
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMinCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin, 但 TF 不是 TestSpec 的注册通路
# (README 四通路为 Kernel/GEIR/ACLNN/E2E), 如需 TF 对标应以 third_party 的 tf
# vendor 形式补充, 本次未做。
# e2e / ONNX / 融合 pass: 均未交付。
