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

"""TTK custom golden for ScatterDiv (kernel mode).

计算公式 (docs/aclnnScatterDiv.md):
    varRef[indices[i], ...] = varRef[indices[i], ...] / updates[i, ...]
多个 updates 命中同一切片时连除（顺序无关，与 TF scatter_div 一致）。
越界索引 (idx < 0 或 idx >= var.shape[0]) 跳过 (scatter_reduce_common_simt.h:112)。
整型 dtype (int32/int8/uint8) 走 C++ 整数除法（向零截断）。
in-place: 输出 var 与输入 var 同缓冲（output_inplace_indexes=(0,)）。

实现说明: 计算由 numpy/python 公式改为 torch 竞品算子拼接（index_select + torch.div +
index_copy_，整型走 rounding_mode="trunc" 即 C++ 向零截断）——纯 numpy 公式实现与被测
kernel 易犯同类错误，会掩盖 kernel 精度短板。numpy 仅保留 I/O 与 dtype 转换。
除法不满足交换律（a/b1/b2 != a/(b1*b2)，逐位不同），故仍按索引顺序逐条下发，
不能折叠成一次 index_reduce；语义（连除顺序、越界跳过、工作 dtype、返回结构）未变。
"""

import numpy as np
import torch

_INT_DTYPES = {"int32", "int8", "uint8"}


def __golden_scatter_div(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtypes = kwargs.get("output_dtypes", None)
    if out_dtypes:
        out_dt_str = out_dtypes[0]
    else:
        out_dt_str = str(var.dtype)
    is_int = out_dt_str in _INT_DTYPES

    var_first = var.shape[0]
    slice_shape = var.shape[1:]
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # work in float32 for fp, int64 for int (avoid overflow during division)
    if is_int:
        work = torch.from_numpy(var.astype(np.int64).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(np.int64).reshape(-1, slice_size))
    else:
        work = torch.from_numpy(var.astype(np.float32).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(np.float32).reshape(-1, slice_size))

    idx_flat = indices.reshape(-1).astype(np.int64)
    n = idx_flat.shape[0]

    for m in range(n):
        idv = int(idx_flat[m])
        if idv < 0 or idv >= var_first:
            continue  # out-of-bound skip
        row = work.index_select(0, torch.tensor([idv]))
        if is_int:
            # C++ integer division (truncation toward zero)
            denom = upd[m : m + 1]
            nonzero = denom != 0
            # match C++ UB conservatively: leave unchanged is not defined;
            # use trunc-toward-zero with denom guarded to avoid a division by zero.
            safe = torch.where(nonzero, denom, torch.ones_like(denom))
            res = torch.where(nonzero, torch.div(row, safe, rounding_mode="trunc"), row)
        else:
            res = torch.div(row, upd[m : m + 1])
        work.index_copy_(0, torch.tensor([idv]), res)

    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }.get(out_dt_str, var.dtype)

    out = work.numpy().reshape(var.shape).astype(np_dtype)
    return [out]


__golden__ = {"kernel": {"scatter_div": "__golden_scatter_div"}}

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


def scatter_div_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_div.
    All the parameters (names and order) follow scatter_div_def.cpp without outputs.
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


class _ScatterDivCompose:
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
        idx, upd = idx[valid], upd[valid]
        # div 无 index_reduce 归约模式: 按索引顺序逐条相除(重复索引累除, 与算子语义一致)。
        # 整型必须走截断除且除数为 0 时保持原值, 不能升 fp32——大 int32 经 float32 会静默丢精度。
        # 整型判定与 golden 同源(_INT_DTYPES = int32/int8/uint8, 即 def 注册的整型面),
        # 不用 dtype.is_floating_point 泛判, 避免与 golden 的分支口径分叉。
        # _INT_DTYPES = {int32, int8, uint8}（def 注册的整型面）的 torch 对应;
        # 不能用 work.numpy() 反查——CUDA 张量转不了 numpy。
        is_int = work.dtype in (torch.int32, torch.int8, torch.uint8)
        for k in range(idx.numel()):
            i = int(idx[k])
            if is_int:
                denom = upd[k]
                nonzero = denom != 0
                safe = torch.where(nonzero, denom, torch.ones_like(denom))
                q = torch.div(work[i], safe, rounding_mode="trunc")
                work[i] = torch.where(nonzero, q, work[i])
            else:
                work[i] = torch.div(work[i], upd[k])
        return [work]


_GOLDEN_FN = __golden_scatter_div


class ScatterDivKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterDivCompose}
    customize_inputs = scatter_div_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_div": "ScatterDivKernelSpec",
    "aclnnScatterDiv": "ScatterDivAclnnSpec",
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


class ScatterDivAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterDivGetWorkspaceSize 的
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
        idx, upd = idx[valid], upd[valid]
        is_int = work.dtype in (torch.int32, torch.int8, torch.uint8)
        for k in range(idx.numel()):
            i = int(idx[k])
            if is_int:
                denom = upd[k]
                nz = denom != 0
                safe = torch.where(nz, denom, torch.ones_like(denom))
                q = torch.div(work[i], safe, rounding_mode="trunc")
                work[i] = torch.where(nz, q, work[i])
            else:
                work[i] = torch.div(work[i], upd[k])
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterDivCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin, 但 TF 不是 TestSpec 的注册通路
# (README 四通路为 Kernel/GEIR/ACLNN/E2E), 如需 TF 对标应以 third_party 的 tf
# vendor 形式补充, 本次未做。
# e2e / ONNX / 融合 pass: 均未交付。
