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
  fp16 输入提升 fp32 计算后回转 fp16，fp32 原生计算；cross_check Promote 场景按
  到达 dtype 透传（fp16 案例到达 fp32 → 按 fp32 计算）。本算子 dtype 仅
  float16/float32，无 bf16，
  不存在 bf16 位桥保真场景。
- third_party：竞品 API torch.nn.functional.normalize 仅经本属性声明，由 TTK 框架在
  独立子进程隔离执行；golden 内不 import / 不调用该竞品 API（aclnn 进程已初始化 aclrt，
  同进程调用竞品框架会 SIGSEGV）。AclnnL2NormalizeTestSpec.golden 中的 torch 仅作
  torch.Tensor <-> numpy 类型桥接（aclnn 比对链要求 golden 为 torch.Tensor），不参与真值计算。
- tolerance：Reduction 计算类算子浮点输出 -> stat_rel_err（2.1 官方枚举，
  以 $TTK/ttk/test_spec/validator.py 为准）。

注册名（与 TTK 消费方式对齐）：
    l2_normalize / L2Normalize -> Kernel / GEIR（CSV op_name；golden 收 numpy.ndarray，
                                   axis/eps 按算子属性名关键字传入）
    aclnnL2Normalize           -> ACLNN（CSV api_name；参数序 = aclnnL2NormalizeGetWorkspaceSize
                                   去掉 workspaceSize/executor：x, axisOptional, eps, out；
                                   axis 亦可经 CSV attributes 键 axis（算子属性名）以 kwargs 注入）
"""

__spec__ = {
    "l2_normalize": "L2NormalizeTestSpec",
    "L2Normalize": "L2NormalizeTestSpec",
    "aclnnL2Normalize": "AclnnL2NormalizeTestSpec",
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

    求和精度：numpy 对非最内层（strided）归约轴走逐元素顺序累加路径，
    fp32 下误差会随归约长度放大；此处归约以 fp64 累加后单次舍回 fp32（真值核口径，
    per-element xc*xc 仍为 fp32、与 kernel 侧逐元素平方完全一致，仅消除求和序误差），
    下游 max/sqrt/div/cast 链路逐位不变。
    """
    if eps is None:
        eps = _EPS_DEFAULT
    eps = float(eps)
    dtype = x_np.dtype
    xc = x_np.astype(numpy.float32) if dtype == numpy.float16 else x_np
    s = numpy.sum(
        xc * xc, axis=_reduce_axes(axis, xc.ndim), keepdims=True, dtype=numpy.float64
    ).astype(numpy.float32)
    y = xc / numpy.sqrt(numpy.maximum(s, eps))
    return y.astype(dtype) if dtype == numpy.float16 else y


class TorchL2NormalizeBenchmark:
    """torch 竞品标杆（spec.reference_oracle：F.normalize p=2.0, dim=axis, eps=sqrt(eps_c)）。

    仅经 third_party 声明、由 TTK 框架在独立子进程执行。axis 可经算子属性名 axis 或
    aclnn 形参名 axisOptional 注入（兼容两种 CSV attributes 键风格）；单元素列表折算为
    int dim（torch dim 契约），多轴以 list 透传。
    """

    def __init__(self, *, axis=None, axisOptional=None, eps=_EPS_DEFAULT, **kwargs):
        self.axis = axis if axis is not None else axisOptional
        self.eps = _EPS_DEFAULT if eps is None else float(eps)

    def __call__(self, x, **kwargs):
        import torch

        # CANN 的 eps 钳在平方和上；映射到 PyTorch（钳在范数上）时取 sqrt(eps)。
        # eps<0 时平方和恒非负，CANN 的 max(sum, eps) 等价于不钳制，故映射为 0。
        # NaN/Inf 则交给 sqrt 与 PyTorch 按 IEEE 语义传播。
        native_eps = 0.0 if self.eps < 0.0 else math.sqrt(self.eps)
        dim = self.axis
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
            normalized = torch.nn.functional.normalize(
                x.unsqueeze(-1), p=2.0, dim=-1, eps=native_eps
            )
            return [normalized.squeeze(-1)]
        if isinstance(dim, (list, tuple)):
            dim = dim[0] if len(dim) == 1 else list(dim)
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


class AclnnL2NormalizeTestSpec:
    """ACLNN 流程（CSV api_name=aclnnL2Normalize）。"""

    def golden(x, axisOptional=None, eps=_EPS_DEFAULT, out=None, **kwargs):
        axis = axisOptional if axisOptional is not None else kwargs.get("axis")
        x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else numpy.asarray(x)
        y_np = _l2_normalize(x_np, axis, eps)
        if hasattr(x, "detach"):
            import torch

            return [torch.from_numpy(y_np)]
        return [y_np]

    third_party = {"torch": TorchL2NormalizeBenchmark}
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }
