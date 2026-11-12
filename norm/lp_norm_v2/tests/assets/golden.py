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

"""LpNormV2 golden：LpNormV2 单算子参考实现。

kernel / GEIR = ``lp_norm_v2``（numpy 入/出）：向量范数
y = max(norm_p(x; axes, keepdim), eps)，其中 norm_p 按 p 取值：
p=0 → 非零元素个数、p=inf → max(|x|)、p=-inf → min(|x|)、
其余 → (sum(|x|^p))^(1/p)；eps 为 dtype 相关的下界（fp16 且 eps<=1e-7 时
夹到 1e-7 / 0，其余取原值）。真值在 FP32（fp16/bf16 输入）或 FP64
（fp32 输入）中完成。

aclnn = ``aclnnNorm`` / ``aclnnLinalgVectorNorm``（torch 入/出）：对应
``torch.norm`` / ``torch.linalg.vector_norm`` 语义，epsilon 恒为 0（无下界）。

e2e = ``torch.linalg.vector_norm``（torch 入/出）：与 torch ATen 向量范数
语义一致，经 torch_npu → aclnnLinalgVectorNorm → LpNormV2 落地 NPU。
"""

import math

import numpy as np
import torch

__spec__ = {
    "lp_norm_v2": "LpNormV2Spec",  # Kernel / GEIR
    "aclnnNorm": "AclnnNormSpec",  # ACLNN
    "aclnnLinalgVectorNorm": "AclnnLinalgVectorNormSpec",  # ACLNN
    "torch.linalg.vector_norm": "TorchLinalgVectorNormSpec",  # E2E
}

_ACL_DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float16,
    27: torch.bfloat16,
}


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.float16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy(tensor, np_dtype):
    """torch CPU 张量 → numpy，按输出 dtype 转换（bf16 经 uint16 位视图）。"""
    np_dtype = np.dtype(np_dtype)
    if np_dtype.name == "bfloat16":
        tensor = tensor.to(torch.bfloat16)
        return tensor.view(torch.float16).numpy().view(np_dtype)
    return np.ascontiguousarray(tensor.numpy().astype(np_dtype, copy=False))


def _compute_dtype(x_dtype):
    """计算精度：fp16/bf16 → fp32；fp32 → fp64。"""
    if x_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return torch.float64


def _effective_epsilon(epsilon, x_dtype):
    """dtype 相关的 eps 下界：仅 fp16 且 eps<=1e-7 时夹到 1e-7（eps≈0 时取 0），
    其余取原值。与 tiling 端 GetAndCheckOtherAttrs 语义一致。"""
    epsilon = float(epsilon)
    if "float16" in str(x_dtype).lower() and epsilon <= 1e-7:
        return 0.0 if math.isclose(epsilon, 0.0, rel_tol=1e-6, abs_tol=0.0) else 1e-7
    return epsilon


def _normalize_dims(axes, ndim):
    """axes → 去重、负轴转正后的归约轴 tuple；空/None 表示全轴归约（返回 None）。"""
    if axes is None or axes == [] or axes == ():
        return None
    if isinstance(axes, int):
        axes = [axes]
    logical_ndim = max(1, ndim)
    out = []
    seen = set()
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += logical_ndim
        if axis not in seen:
            seen.add(axis)
            out.append(axis)
    return tuple(out)


def _vector_norm(x_t, p, dims, keepdim):
    """向量范数核心（torch 张量入/出），dims=None 表示全轴归约。

    torch.linalg.vector_norm 对 p=0/inf/-inf/任意实数均有对应语义；仅在
    inf/-inf 且归约轴为空的退化场景（torch 不支持）手工填充数学恒等值。
    """
    p = float(p)
    if math.isinf(p) and dims is not None:
        for axis in dims:
            if x_t.shape[axis] == 0:
                fill = float("-inf") if p > 0 else float("inf")
                if keepdim:
                    shape = [1 if i in dims else s for i, s in enumerate(x_t.shape)]
                else:
                    shape = [s for i, s in enumerate(x_t.shape) if i not in dims]
                return torch.full(shape, fill, dtype=x_t.dtype)
    return torch.linalg.vector_norm(x_t, ord=p, dim=dims, keepdim=keepdim)


def lp_norm_v2_golden(x, *, p=2.0, axes=[], keepdim=False, epsilon=1e-12, **kwargs):
    """Kernel/GEIR golden：numpy 入/出，返回 [y]。

    参数名与顺序跟随 lp_norm_v2_def.cpp 的输入与属性（不含输出）。
    """
    x_t = _np_to_torch(x)
    dims = _normalize_dims(axes, x_t.dim())
    x_c = x_t.to(_compute_dtype(x_t.dtype))
    res = _vector_norm(x_c, p, dims, bool(keepdim))
    eps = _effective_epsilon(epsilon, x_t.dtype)
    res = torch.maximum(res, torch.tensor(eps, dtype=x_c.dtype))
    return [_torch_to_numpy(res, x.dtype)]


def aclnn_norm_golden(self_t, p, dim, keepdim, out=None, **kwargs):
    """ACLNN golden（aclnnNorm）：torch 入/出，返回 [y]。

    参数位置跟随 aclnnNormGetWorkspaceSize（不含 workspaceSize 与 executor）。
    """
    dims = _normalize_dims(dim, self_t.dim())
    x_c = self_t.to(_compute_dtype(self_t.dtype))
    res = _vector_norm(x_c, float(p), dims, bool(keepdim))
    return [res.to(self_t.dtype)]


def aclnn_linalg_vector_norm_golden(
    self_t, ord, dims, keepDims, dtype, out=None, **kwargs
):
    """ACLNN golden（aclnnLinalgVectorNorm）：torch 入/出，返回 [y]。

    参数位置跟随 aclnnLinalgVectorNormGetWorkspaceSize（不含 workspaceSize 与
    executor）；dtype 为 aclDataType（0=float32/1=float16/27=bfloat16），
    指定计算与输出精度。
    """
    dims = _normalize_dims(dims, self_t.dim())
    compute_dtype = _ACL_DTYPE_TO_TORCH.get(int(dtype), torch.float32)
    res = torch.linalg.vector_norm(
        self_t.to(compute_dtype), ord=float(ord), dim=dims, keepdim=bool(keepDims)
    )
    return [res]


def linalg_vector_norm_golden(
    x, ord=2, dim=None, keepdim=False, dtype=None, out=None, **kwargs
):
    """E2E golden（torch.linalg.vector_norm）：torch 入/出，返回 [y]。

    参数跟随 torch.linalg.vector_norm 签名；dtype 缺省时按输入 dtype 计算。
    """
    dims = _normalize_dims(dim, x.dim())
    x_c = x.to(_compute_dtype(x.dtype))
    res = _vector_norm(x_c, ord, dims, bool(keepdim))
    return [res.to(x.dtype)]


class LpNormV2XpuRef:
    """torch 三方标杆（GPU 参考，kernel/GEIR 交叉比对）。

    输入为设备侧 torch.Tensor：与 torch 向量范数语义一致（无 eps 下界），
    按输入 dtype 计算（与三方 GPU 同精度）。
    """

    def __call__(self, x, *, p=2.0, axes=[], keepdim=False, epsilon=1e-12, **kwargs):
        dims = _normalize_dims(axes, x.dim())
        res = _vector_norm(x, float(p), dims, bool(keepdim))
        return [res.to(x.dtype)]


class AclnnNormXpuRef:
    """torch 三方标杆（aclnnNorm 交叉比对）。"""

    def __call__(self, self_t, p, dim, keepdim, out=None, **kwargs):
        dims = _normalize_dims(dim, self_t.dim())
        res = _vector_norm(self_t, float(p), dims, bool(keepdim))
        return [res.to(self_t.dtype)]


class AclnnLinalgVectorNormXpuRef:
    """torch 三方标杆（aclnnLinalgVectorNorm 交叉比对）。"""

    def __call__(self, self_t, ord, dims, keepDims, dtype, out=None, **kwargs):
        dims = _normalize_dims(dims, self_t.dim())
        compute_dtype = _ACL_DTYPE_TO_TORCH.get(int(dtype), torch.float32)
        res = torch.linalg.vector_norm(
            self_t.to(compute_dtype), ord=float(ord), dim=dims, keepdim=bool(keepDims)
        )
        return [res]


class TorchLinalgVectorNormXpuRef:
    """torch 三方标杆（e2e 交叉比对）。"""

    def __call__(
        self, x, ord=2, dim=None, keepdim=False, dtype=None, out=None, **kwargs
    ):
        dims = _normalize_dims(dim, x.dim())
        res = _vector_norm(x, float(ord), dims, bool(keepdim))
        return [res.to(x.dtype)]


class LpNormV2Spec:
    """kernel/GEIR 流程：numpy 入/出，单输出。"""

    golden = staticmethod(lp_norm_v2_golden)
    third_party = {"torch": LpNormV2XpuRef}


class AclnnNormSpec:
    """aclnn 流程（aclnnNorm）：torch 入/出，单输出。"""

    golden = staticmethod(aclnn_norm_golden)
    third_party = {"torch": AclnnNormXpuRef}


class AclnnLinalgVectorNormSpec:
    """aclnn 流程（aclnnLinalgVectorNorm）：torch 入/出，单输出。"""

    golden = staticmethod(aclnn_linalg_vector_norm_golden)
    third_party = {"torch": AclnnLinalgVectorNormXpuRef}


class TorchLinalgVectorNormSpec:
    """e2e 流程：torch 入/出，单输出。"""

    golden = staticmethod(linalg_vector_norm_golden)
    third_party = {"torch": TorchLinalgVectorNormXpuRef}
