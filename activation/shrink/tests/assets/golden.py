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

"""Shrink golden（kernel / GEIR / aclnn 通路，TestSpec 格式）。

ONNX Shrink（ai.onnx::Shrink）非线性收缩，按输入与阈值 lambd 的关系分三区：
    y = x - bias  (x >  lambd)
    y = x + bias  (x < -lambd)
    y = 0         (-lambd <= x <= lambd)

Canonical IO order (shrink_def.cpp):
    inputs : input_x (fp16/fp32, ND)
    outputs: output_y (同 input_x shape/dtype)
    attrs  : lambd (float, 默认 0.5), bias (float, 默认 0.0)

aclnn 通路参数序照 aclnnShrinkGetWorkspaceSize(self, lambd, bias, out)，
golden 签名去掉 workspaceSize/executor。

分支语义与 arch35 内核逐位对齐：
    - lambd/bias 由 host float32 经 static_cast<T> 进入内核比较与加减，
      golden 同样 cast 到输入 dtype 后再比较/加减，阈值边界与内核一致
      （如 lambd=0.3 的 fp16 表示 0.30004883 大于真值，浮点原位比较会把
      x=fp16(0.30004883) 误判入正区域，cast 后与内核同为死区）。
    - kernel/GEIR 通路 host tiling 把 lambd<0 钳位为 0（shrink_tiling_arch35.cpp
      ``tiling->lambd = (lambd < 0.0f) ? 0.0f : lambd``），golden 同步钳位；
      aclnn 通路在接口层拒绝 lambd<0（CheckLambdValue），负值不会进入计算。
    - 元素级 select/add-sub 单步舍入、无累加，native dtype 即精确真值；
      Promote 档输入被抬升后按抬升精度计算（cast 跟随输入 dtype 自动成立）。

精度策略：
    golden / third_party —— 均按输入 dtype 原位计算（内核 fp16 通路即 fp16
        算术，无 widening），出口保持原 dtype。
    入参/出口约定 —— 四入口统一经 _to_cpu_tensor 归一化（torch →
        detach().cpu()，numpy → from_numpy）；golden 出口统一 numpy.ndarray，
        三方标杆出口统一 torch.Tensor（XPU 服务端自行转换）。
    写法独立 —— golden 用嵌套 where，third_party 用符号分解
        （|x|>lambd 时 x-sign(x)*bias），两套写法互为对照，cross_check
        三方误差不再与 golden 同源恒 0；dtype cast/lambd 钳位等边界语义
        为设备契约镜像（非写法独立范围）。
    tolerance   —— cross_check L1（对齐 operator-artifacts 参考档）。
"""

import numpy
import torch


# 双拼写注册：opInterface 名（安装包 op-info 键，TTK kernel/geir 通路按 CSV op_name
# 精确匹配）与 GE 算子类型名（生成侧 op_name 拼写）均可直接解析，无需执行副本归一。
__spec__ = {
    "shrink": "ShrinkKernelTestSpec",
    "Shrink": "ShrinkKernelTestSpec",
    "aclnnShrink": "AclnnShrinkTestSpec",
}

__golden__ = {
    "kernel": {
        "shrink": "shrink_golden",
        "Shrink": "shrink_golden",
    },
    "aclnn": {
        "aclnnShrink": "aclnn_shrink_golden",
    },
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _to_cpu_tensor(x):
    """入参归一化：torch.Tensor → detach().cpu()；其余（numpy 等）→ from_numpy。

    两条通路（kernel/aclnn）与 golden/三方标杆统一复用，兼容 TTK 以
    numpy.ndarray 传入的 host 侧惯例，避免 numpy.dtype 直入 torch 语义。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.from_numpy(numpy.asarray(x))


def _shrink_where(x_t, lambd, bias):
    """按 arch35 内核分支语义计算。

    lambd/bias 先 cast 到输入 dtype（内核 static_cast<T>），比较与加减均在
    该 dtype 下进行，保证阈值边界与内核逐位一致；NaN 输入两侧比较均为假，
    与内核 Compares 双 mask 落空同走死区 0。
    """
    lambd_t = torch.as_tensor(lambd, dtype=x_t.dtype, device=x_t.device)
    bias_t = torch.as_tensor(bias, dtype=x_t.dtype, device=x_t.device)
    return torch.where(
        x_t > lambd_t,
        x_t - bias_t,
        torch.where(x_t < -lambd_t, x_t + bias_t, torch.zeros_like(x_t)),
    )


def _clamp_lambd(lambd):
    """kernel/GEIR 通路 host tiling 对 lambd<0 钳位为 0，golden 镜像该行为。"""
    return max(float(lambd), 0.0)


def _shrink_sign_decomposed(x_t, lambd, bias):
    """三方标杆的独立写法实现（区别于 golden 的嵌套 where）。

    三区语义改用符号分解：|x| > lambd 时 y = x - sign(x)*bias
    （正区即 x-bias、负区即 x+bias），否则死区 0。与嵌套 where 的写法
    独立，golden 若存在公式转写错误，cross_check 三方误差不再恒为 0。

    语义契约仍与设备一致（非写法层面）：lambd/bias cast 到输入 dtype
    （内核 static_cast<T>，有 tiling/README 依据）；NaN 两侧比较均为假
    落入死区 0（与内核 Compares 双 mask 落空一致）；lambd 经调用方按通路
    钳位后恒非负，故 |x|>lambd 与公开公式的三区划分等价。
    """
    lambd_t = torch.as_tensor(lambd, dtype=x_t.dtype, device=x_t.device)
    bias_t = torch.as_tensor(bias, dtype=x_t.dtype, device=x_t.device)
    outside = x_t.abs() > lambd_t
    return torch.where(outside, x_t - torch.sign(x_t) * bias_t, torch.zeros_like(x_t))


def shrink_golden(input_x, lambd=0.5, bias=0.0, *args, **kwargs):
    """kernel/GEIR 通路 golden（签名照 shrink_def.cpp：输入在前，属性随后）。

    出口统一为 numpy（与 aclnn 通路一致）。
    """
    del args
    result = _shrink_where(_to_cpu_tensor(input_x), _clamp_lambd(lambd), bias)
    return [result.numpy()]


def aclnn_shrink_golden(self, lambd, bias, out=None, **kwargs):
    """aclnn 通路 golden（签名照 aclnnShrinkGetWorkspaceSize，去掉 workspaceSize/executor）。

    lambd<0 在接口层已被 CheckLambdValue 拒绝，合法入参不含负值，此处不做钳位。
    入参归一化与出口 numpy 与 kernel 通路一致（torch/numpy 入参均可，
    避免 numpy.ndarray 直入 torch 语义抛 TypeError）。
    """
    del out
    result = _shrink_where(_to_cpu_tensor(self), lambd, bias)
    return [result.numpy()]


class ShrinkKernelReference:
    """kernel/GEIR 三方标杆：符号分解写法（独立于 golden 的嵌套 where）。

    镜像内核 native-dtype 算术路径与 host tiling 的 lambd<0 钳位。
    三方腿出口保持 torch.Tensor（TTK XPU 服务端自行做 numpy 转换）。
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input_x, lambd=0.5, bias=0.0, **kwargs):
        return [
            _shrink_sign_decomposed(_to_cpu_tensor(input_x), _clamp_lambd(lambd), bias)
        ]


class AclnnShrinkReference:
    """aclnn 三方标杆：参数名照 aclnnShrinkGetWorkspaceSize，符号分解写法。

    ``__call__`` 形参须按 API 参数名显式声明（TTK XPU 服务端按名绑定）：
    绑定方法的 ``**kwargs`` 收不到名为 ``self`` 的输入（与实例槽冲突保护），
    ``self`` 必须是绑定签名里的真实形参才能按名绑定。lambd<0 已被接口层
    拒绝，不做钳位。三方腿出口保持 torch.Tensor（与 kernel 标杆一致）。
    """

    def __init__(self, **kwargs):
        pass

    def __call__(_impl, self, lambd=0.5, bias=0.0, out=None, **kwargs):
        del out
        return [_shrink_sign_decomposed(_to_cpu_tensor(self), lambd, bias)]


class ShrinkKernelTestSpec:
    """kernel / GEIR 通路 TestSpec（CSV op_name 双拼写注册共用）。"""

    golden = shrink_golden
    third_party = {"torch": ShrinkKernelReference}
    tolerance = _TOL


class AclnnShrinkTestSpec:
    """aclnn 通路 TestSpec（CSV api_name 注册）。"""

    golden = aclnn_shrink_golden
    third_party = {"torch": AclnnShrinkReference}
    tolerance = _TOL
