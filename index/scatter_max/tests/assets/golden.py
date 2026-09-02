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

"""scatter_max TTK 自定义 golden plugin（kernel 模式）。

计算公式（aclnnScatterMax.md 计算公式节 / proto.h:36）:
  varRef[indices[i], ...] = max(varRef[indices[i], ...], updates[i, ...])
  - 多个 updates 作用到 var 同一切片时依次取最大值（顺序无关）。
  - shape 约束: updates.shape = indices.shape + var.shape[1:]。
  - indices 越界值 (idx < 0 或 idx >= var.shape[0]) 被算子跳过（kernel simt.h:112）。
  - var 原地更新，输出即更新后的 var。

输入顺序: (var, indices, updates)  输出: [var]

实现说明: 计算由 numpy 逐条 np.maximum 改为 torch 竞品算子 Tensor.index_reduce_(reduce="amax",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误, 会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义(重复索引取最大、越界跳过、工作 dtype、返回结构) 未变。
"""

import numpy as np
import torch


def __golden_scatter_max(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out = var.copy()
    var_first_dim = out.shape[0] if out.ndim >= 1 else 0

    idx_flat = indices.reshape(-1)
    n_idx = idx_flat.shape[0]

    if n_idx == 0 or var_first_dim == 0 or updates.size == 0:
        return [out.astype(var.dtype)]

    # updates 展平为 (n_idx, *slice_shape)：slice_shape = var.shape[1:]
    slice_shape = out.shape[1:]
    upd = updates.reshape((n_idx,) + tuple(slice_shape))

    # 浮点用 float32 中间计算（fp16 numpy 不自动提升）；整型保持原类型精确比对。
    # 注意：max 不累加、永不溢出，整型绝不能转 float32——大 int32(>2^24) 经 float32
    # round-trip 会丢精度，导致与 kernel 的精确 int32 max 不符（曾误报 inf/大值用例失败）。
    is_float = np.issubdtype(var.dtype, np.floating)
    work_dtype = np.float32 if is_float else var.dtype
    work = torch.from_numpy(out.astype(work_dtype))
    upd_w = torch.from_numpy(upd.astype(work_dtype))

    # 越界索引先剔除（与 kernel 的 skip 一致），剩下的交给 torch 竞品算子；
    # include_self=True 即 max(var 原值, 命中该行的所有 updates)，重复索引顺序无关。
    idx_t = torch.from_numpy(idx_flat.astype(np.int64))
    valid = (idx_t >= 0) & (idx_t < var_first_dim)
    idx_t = idx_t[valid]
    if idx_t.numel() > 0:
        work.index_reduce_(0, idx_t, upd_w[valid], "amax", include_self=True)

    return [work.numpy().astype(var.dtype)]


__golden__ = {"kernel": {"scatter_max": "__golden_scatter_max"}}

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


def scatter_max_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_max.
    All the parameters (names and order) follow scatter_max_def.cpp without outputs.
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


class _ScatterMaxCompose:
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
            work = work.index_reduce(0, idx, upd[valid], "amax", include_self=True)
        return [work]


_GOLDEN_FN = __golden_scatter_max


class ScatterMaxKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterMaxCompose}
    customize_inputs = scatter_max_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_max": "ScatterMaxKernelSpec",
    "aclnnScatterMax": "ScatterMaxAclnnSpec",
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


class ScatterMaxAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterMaxGetWorkspaceSize 的
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
            work = work.index_reduce(0, idx, upd[valid], "amax", include_self=True)
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMaxCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin, 但 TF 不是 TestSpec 的注册通路
# (README 四通路为 Kernel/GEIR/ACLNN/E2E), 如需 TF 对标应以 third_party 的 tf
# vendor 形式补充, 本次未做。
# e2e / ONNX / 融合 pass: 均未交付。
