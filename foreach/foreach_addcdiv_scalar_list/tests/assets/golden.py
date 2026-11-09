#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_addcdiv_scalar_list 三方 Golden — 标量列表除加。

公式: y[i] = x1[i] + scalars[i] * (x2[i] / x3[i])，scalars[i] 为第 i 个
tensor 的标量系数。支持 fp16/fp32/bf16。

实现: torch._foreach_div 做除法 + Python 算术乘加——对齐设备端求值序
(CUDA: fma(alpha, t1/t2, input)，先除后乘加)。不直接调用
torch._foreach_addcdiv: CPU 后端按乘先结合序(self + val*t1/t2 左结合)
求值，在 |s*x2| 溢出而 |s*(x2/x3)| 可表示的角点产出 inf，与 GPU/NPU
设备端除先序(fma(alpha, t1/t2, input))分叉。乘加为逐算子两次舍入，
与设备端 fma 单次舍入的差异由比对容差吸收。纯函数语义: fp16/bf16
输入提升至 fp32 完成除法与乘加，末尾单次舍回原生 dtype——对齐设备
kernel 的 float32 中间计算(商不提前舍回，商溢出而结果有限的角点可
正确表达); fp32/fp64 原精度直算。输出 dtype 与收到输入的 dtype
一致。

- 不读取任何 TTK 模式字段（golden_mode 等）: 调用方抬升输入精度时，
  golden 在抬升后的精度直出（如收到 fp64 即 fp64 直出），输出量化的
  对齐职责属比对器/容差设计。
- kernel 路（numpy）经 torch 张量桥接计算后转回 numpy; bf16 无 numpy
  原生 dtype, 经 uint16 位视图携带位型投递（不经 f32 中转）。
"""

__spec__ = {
    "foreach_addcdiv_scalar_list": "ForeachAddcdivScalarListKernelSpec",
    "ForeachAddcdivScalarList": "ForeachAddcdivScalarListKernelSpec",
    "aclnnForeachAddcdivScalarList": "AclnnForeachAddcdivScalarListSpec",
    "torch._foreach_addcdiv": "TorchForeachAddcdivScalarListSpec",
    "torch._foreach_addcdiv.ScalarList": "TorchForeachAddcdivScalarListSpec",
}

import numpy as np

# 精度标准: 计算类浮点走三方交叉比对(cross_check); level 为 cross_check
# 比对预设档位注记, 逐 dtype 数值容差由 TTK cross_check L1 预设与算子
# spec.yaml numerical_tolerance 承载, golden 侧不展开。
_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_scalar(v):
    import torch

    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


def _to_scalars(v):
    import torch

    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.reshape(-1).tolist()
    return [_to_scalar(t) for t in v]


def _np_to_torch(t):
    """numpy -> torch（bf16 经 uint16 位视图携带位型投递，不经 f32 中转）。"""
    import torch

    arr = np.ascontiguousarray(t)
    if arr.dtype.name == "bfloat16":
        return (
            torch.from_numpy(arr.view("uint16")).view(torch.bfloat16).reshape(t.shape)
        )
    return torch.from_numpy(arr).reshape(t.shape)


def _to_numpy(r, t):
    """torch 输出 -> numpy（bf16 按输入数组自带 dtype 经 uint16 位视图还原位型）。"""
    import torch

    if r.dtype == torch.bfloat16:
        return r.view(torch.uint16).numpy().view(np.asarray(t).dtype)
    return r.numpy()


def _foreach_addcdiv_divfirst(x1, x2, x3, scalars):
    """div-first 计算: q = x2/x3 先除(torch._foreach_div), 再 Python 算术
    乘加 a + s*q——保留除先序(CUDA: fma(alpha, t1/t2, input), 先除后
    乘加)。fp16/bf16 输入提升至 fp32 完成除法与乘加，末尾单次舍回原生
    dtype——对齐设备 kernel 的 float32 中间计算(商不提前舍回，商溢出
    而结果有限的角点可正确表达); fp32/fp64 原精度直算。空列表早退
    (torch._foreach_div 拒绝空 TensorList); scalars 归一为标量列表
    (0-d/1-d 张量与裸列表统一)。"""
    import torch

    if len(x1) == 0:
        return []
    scalars_list = _to_scalars(scalars)
    dt = x1[0].dtype
    if dt in (torch.float16, torch.bfloat16):
        x1f = [t.float() for t in x1]
        x2f = [t.float() for t in x2]
        x3f = [t.float() for t in x3]
        q = torch._foreach_div(x2f, x3f)
        return [r.to(dt) for r in [a + s * b for a, b, s in zip(x1f, q, scalars_list)]]
    q = torch._foreach_div(x2, x3)
    return [a + s * b for a, b, s in zip(x1, q, scalars_list)]


class _TorchForeachAddcdivScalarListImpl:
    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, x1, x2, x3, **kwargs):
        import torch

        return list(torch._foreach_addcdiv(x1, x2, x3, self._scalars))


class _TorchForeachAddcdivScalarListE2EImpl:
    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, *args, scalars=None):
        import torch

        return list(torch._foreach_addcdiv(args[0], args[1], args[2], self._scalars))


_THIRD_PARTY = {"torch": _TorchForeachAddcdivScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachAddcdivScalarListE2EImpl}


class ForeachAddcdivScalarListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。div-first 计算
    (torch._foreach_div + Python 乘加)，输出 dtype 与收到输入一致。"""

    def golden(x1, x2, x3, scalars, **kwargs):
        # reshape(t.shape): 0-dim 输入经 ascontiguousarray 会升为 (1,)，此处恢复原 shape
        x1l = [_np_to_torch(t) for t in x1]
        x2l = [_np_to_torch(t) for t in x2]
        x3l = [_np_to_torch(t) for t in x3]
        y = _foreach_addcdiv_divfirst(x1l, x2l, x3l, scalars)
        return [_to_numpy(r, t) for r, t in zip(y, x1)]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachAddcdivScalarListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。div-first 计算
    (torch._foreach_div + Python 乘加)，输出 dtype 与收到输入一致。"""

    def golden(x1, x2, x3, scalars, out=None, **kwargs):
        # out is read-only: the NPU kernel writes its output there after this
        # golden returns; writing here would mask NPU execution failures.
        return _foreach_addcdiv_divfirst(x1, x2, x3, scalars)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachAddcdivScalarListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。与其他通路同一
    纯函数语义：div-first 计算(torch._foreach_div + Python 乘加)，输出
    dtype 与收到输入一致。"""

    def golden(x1, x2, x3, scalars, **kwargs):
        return _foreach_addcdiv_divfirst(x1, x2, x3, scalars)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
