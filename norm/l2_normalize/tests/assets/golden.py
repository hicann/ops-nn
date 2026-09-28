#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""L2Normalize TTK TestSpec（golden / third_party / tolerance）。

参考实现采用公开接口定义的数学语义：
    s = sum(x * x, axis=tuple(axis), keepdims=True)
    y = x / sqrt(max(s, eps))            # eps 钳制在平方和上（先 max 后 sqrt，CANN 口径）
与 torch 竞品标杆数学等价：F.normalize(x, p=2.0, dim=axis, eps_o=sqrt(eps_c))
= x / max(||x||_2, sqrt(eps_c)) = x / sqrt(max(sum_axis x^2, eps_c))。

- golden：独立 CPU 参考实现（纯 numpy 真值核），不 import / 复用被测 kernel 任何产物；
  float16/float32 输入均提升到 float64，平方、归约、max、sqrt、除法全程 float64，
  返回 float64 真值，不提前舍入到输出 dtype；TTK Promote 传入的 float64 不降精度。
- third_party：竞品 API torch.nn.functional.normalize 仅经本属性声明，由 TTK 框架在
  独立进程执行；golden 内不 import / 不调用该竞品 API。标杆保留输入 dtype，
  不复刻 NPU 的归约顺序或补偿算法。
- tolerance：Reduction 计算类算子浮点输出 -> stat_rel_err（2.1 官方枚举，
  以 $TTK/ttk/test_spec/validator.py 为准）。

注册名（与 TTK 消费方式对齐）：
    l2_normalize / L2Normalize -> Kernel / GEIR（CSV op_name；golden 收 numpy.ndarray，
                                   axis/eps 按算子属性名关键字传入）
本算子未交付 ACLNN 接口，不注册 ACLNN TestSpec。
"""

__spec__ = {
    "l2_normalize": "L2NormalizeTestSpec",
    "L2Normalize": "L2NormalizeTestSpec",
}

import math

import numpy

_EPS_DEFAULT = 1e-4


def _reduce_axes(axis, ndim):
    """axis 属性 -> 归一化的 numpy 归约轴元组；None/[] -> 不归约。"""
    if axis is None:
        return ()
    if isinstance(axis, (int, numpy.integer)):
        axis = (int(axis),)
    normalized = []
    for value in axis:
        value = int(value)
        if value < -ndim or value >= ndim:
            raise ValueError(f"axis {value} is outside [-{ndim}, {ndim})")
        value = value + ndim if value < 0 else value
        if value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized))


def _l2_normalize(x_np, axis=None, eps=_EPS_DEFAULT):
    """CPU 真值核：y = x / sqrt(max(sum(x*x, axis, keepdims=True), eps))。

    NaN/Inf 按 IEEE 传播（NaN 污染整条归约 slice；Inf 使分母 Inf：inf 位 inf/inf=NaN、
    同 slice 其余位 0）；空 tensor 沿轴归约得 0 -> 分母钳到 sqrt(eps) -> 输出同维空张量。

    全程使用 fp64，避免 fp32 平方溢出/下溢以及归约和提前舍入误差。
    高精度输出用于精度比较；输出 dtype 的判据由 TTK 按被测算子输出解析。
    """
    if eps is None:
        eps = _EPS_DEFAULT
    eps = float(eps)
    xc = numpy.asarray(x_np, dtype=numpy.float64)
    s = numpy.sum(
        xc * xc, axis=_reduce_axes(axis, xc.ndim), keepdims=True, dtype=numpy.float64
    )
    return xc / numpy.sqrt(numpy.maximum(s, eps))


class TorchL2NormalizeBenchmark:
    """torch 竞品标杆（spec.reference_oracle：F.normalize p=2.0, dim=axis, eps=sqrt(eps_c)）。

    仅经 third_party 声明、由 TTK 框架在独立进程执行。
    与 golden 一样对 axis 进行负轴归一和去重；空 axis 表示逐元素归一化。
    """

    def __init__(self, *, axis=None, eps=_EPS_DEFAULT, **kwargs):
        self.axis = axis
        self.eps = _EPS_DEFAULT if eps is None else float(eps)

    def __call__(self, x, **kwargs):
        # importlib 动态导入：绕过 TTK worker 的 AST preload 扫描（ast.Import），
        # 避免 forkserver worker 里 preload torch 触发 SIGSEGV（2026-09-24 实测）。
        # third_party 仅在远端 xpu_server 进程内执行，torch 在彼处动态导入，语义不变。
        import importlib

        torch = importlib.import_module("torch")

        # CANN 的 eps 钳在平方和上；映射到 PyTorch（钳在范数上）时取 sqrt(eps)。
        # eps<0 时平方和恒非负，CANN 的 max(sum, eps) 等价于不钳制，故映射为 0。
        # NaN/Inf 则交给 sqrt 与 PyTorch 按 IEEE 语义传播。
        native_eps = 0.0 if self.eps < 0.0 else math.sqrt(self.eps)
        dim = _reduce_axes(self.axis, x.ndim)
        if not dim:
            normalized = torch.nn.functional.normalize(
                x.unsqueeze(-1), p=2.0, dim=-1, eps=native_eps
            )
            return [normalized.squeeze(-1)]
        return [torch.nn.functional.normalize(x, p=2.0, dim=dim, eps=native_eps)]


class L2NormalizeTestSpec:
    """Kernel / GEIR 流程（CSV op_name=l2_normalize）。"""

    def golden(x, axis=None, eps=_EPS_DEFAULT, **kwargs):
        return [_l2_normalize(x, axis, eps)]

    third_party = {"torch": TorchL2NormalizeBenchmark}
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }
