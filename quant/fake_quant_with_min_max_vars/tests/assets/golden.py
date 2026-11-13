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

"""TTK 多通路 golden + third_party for fake_quant_with_min_max_vars.

两条通路（geir/kernel + tf E2E）各含独立 golden + third_party + tolerance。
本算子 CMakeLists 标注 ACLNNTYPE aclnn_exclude（无 aclnn API），故无 aclnn 通路。

与 fake_quant_with_min_max_args 的区别：
  - min/max 是 tensor 输入（可训练变量，shape (1,)），不是标量属性。
  - 仅 2 个属性：num_bits / narrow_range（无 min/max 属性）。

竞品与精度策略：
  - 唯一真实竞品为 TensorFlow ``tf.quantization.fake_quant_with_min_max_vars``（REQUIREMENTS §3 / DESIGN §10.1）。
  - TF API 要求 min/max 为标量（rank 0），本算子 min/max 为 shape (1,) tensor，
    third_party 用 class 形式在 __call__ 内取标量后调 TF。
  - DESIGN §6 明确记录精度分歧：元素级量化用 round-half-up（floor(x+0.5)），TF 用
    round-half-away-from-zero；负数 0.5 边界处 1 量化级别偏移。故 golden 用 torch 小算子
    拼接，与 kernel 计算链逐行对齐：
    * Nudge zp: round-half-away（与 TF 一致，DESIGN §6 表「Nudge zero point」）
    * 元素级量化: round-half-up = floor(x+0.5)（对齐 kernel CAST_FLOOR，DESIGN §7）
    * 退化: min==max → 输出全零（DESIGN §7.4）
  - dtype 仅 float32（spec.yaml supported_combinations）。
  - 精度标准: cross_check L1（mare≤5 / mere≤1.5 / rmse≤1.5）。
"""

import math

import numpy as np

# 注意：torch 必须延迟导入——xpu-server 的 tf 端点（跑 _TfCompetitor）无 torch，
# 模块级 import 会让整个 spec 模块加载失败；torch 仅在 golden 腿（kernel/E2E golden）
# 执行时需要，均在函数内导入。

__spec__ = {
    # geir / kernel 通路：ttk kernel | GEIR，CSV op_name 查找，golden 收 numpy
    "fake_quant_with_min_max_vars": "FakeQuantWithMinMaxVarsKernelSpec",
    "FakeQuantWithMinMaxVars": "FakeQuantWithMinMaxVarsKernelSpec",
    # E2E 通路：ttk e2e，CSV api_name = tf 竞品 API 全路径，golden 收 torch.Tensor
    "tf.quantization.fake_quant_with_min_max_vars": "TfFakeQuantWithMinMaxVarsSpec",
}


# ─── Nudge 预计算（kernel 端，min/max 从 tensor 读取）──────────────────


def _round_half_away(z):
    """std::round 语义：round half away from zero（Nudge zp 用，与 TF 一致）."""
    if z >= 0:
        return math.floor(z + 0.5)
    return math.ceil(z - 0.5)


def _nudge(min_v, max_v, num_bits, narrow_range):
    """返回 (nudged_min, nudged_max, scale, inv_scale)。min==max 时返回 None 表示退化."""
    if min_v == max_v:
        return None  # 退化：输出全零
    q_min = 1 if narrow_range else 0
    q_max = 2**num_bits - 1
    scale = np.float32(max_v - min_v) / np.float32(q_max - q_min)
    inv_scale = np.float32(q_max - q_min) / np.float32(max_v - min_v)  # 独立除法
    zp = np.float32(q_min) - np.float32(min_v) / scale
    if zp <= q_min:
        nudged_zp = float(q_min)
    elif zp >= q_max:
        nudged_zp = float(q_max)
    else:
        nudged_zp = float(_round_half_away(float(zp)))
    nudged_min = float(np.float32(q_min - nudged_zp) * scale)
    nudged_max = float(np.float32(q_max - nudged_zp) * scale)
    return nudged_min, nudged_max, scale, inv_scale


# ─── 前向计算（kernel DESIGN §7 逐点对齐）─────────────────────────────


def _forward(x, min_v, max_v, num_bits, narrow_range):
    """前向：Nudge → clamp → round-half-up 量化 → 乘 scale 反量化.

    退化（DESIGN §7.4）：min==max → 输出全零。
    元素级量化用 round-half-up = floor(x+0.5)（对齐 kernel CAST_FLOOR，DESIGN §6/§7）。
    """
    import torch

    num_bits = 8 if num_bits is None else int(num_bits)
    narrow_range = False if narrow_range is None else bool(narrow_range)
    nudge = _nudge(min_v, max_v, num_bits, narrow_range)
    if nudge is None:
        return torch.zeros_like(x)
    nudged_min, nudged_max, scale, inv_scale = nudge
    clamped = torch.clamp(x, nudged_min, nudged_max)
    # round-half-up: floor(x + 0.5)
    y = torch.floor(clamped * inv_scale + 0.5) * scale
    return y


# ─── third_party TF 竞品（class form，处理 min/max 标量转换）─────────
# 本算子唯一真实竞品为 TensorFlow tf.quantization.fake_quant_with_min_max_vars
# （REQUIREMENTS §3 / DESIGN §10.1 / spec.yaml framework: tensorflow）。
# TF API 要求 min/max 为标量（rank 0），而本算子 min/max 是 shape (1,) tensor，
# 故在 __call__ 内取标量后调用 TF。server 侧 import tensorflow（standalone）。


class _TfCompetitor:
    """third_party tf — 调用 tf.quantization.fake_quant_with_min_max_vars.

    __init__ 收属性，__call__ 收 x/min/max tensor，把 min/max 从 shape (1,)
    取标量后调 TF API（TF 要求 rank 0）。
    """

    def __init__(self, *, num_bits=8, narrow_range=False, **kwargs):
        self.num_bits = 8 if num_bits is None else int(num_bits)
        self.narrow_range = False if narrow_range is None else bool(narrow_range)

    def __call__(self, x, min, max, **kwargs):
        import tensorflow as tf

        min_v = tf.reshape(min, [])
        max_v = tf.reshape(max, [])
        return [
            tf.quantization.fake_quant_with_min_max_vars(
                x, min_v, max_v, num_bits=self.num_bits, narrow_range=self.narrow_range
            )
        ]


class _TfCompetitorE2E(_TfCompetitor):
    """third_party tf - E2E 通路：XPU服务端按名绑定，参数对齐 TF API 签名 inputs/min/max."""

    def __call__(self, inputs, min, max, **kwargs):
        return super().__call__(inputs, min, max)


# ─── 通路 1: geir / kernel (GEIR) ─────────────────────────────────
# golden 收 numpy.ndarray，third_party 收 torch.Tensor
# GEIR 复用 kernel 的 spec。


class FakeQuantWithMinMaxVarsKernelSpec:
    """geir / kernel 通路 — golden 收 numpy.ndarray."""

    def golden(x, min, max, num_bits=8, narrow_range=False, **kwargs):
        import torch

        x_t = torch.as_tensor(x)
        min_v = float(torch.as_tensor(min).reshape(-1)[0])
        max_v = float(torch.as_tensor(max).reshape(-1)[0])
        y = _forward(x_t, min_v, max_v, num_bits, narrow_range)
        return [y.numpy()]

    third_party = {
        "tf": _TfCompetitor,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }


# ─── 通路 2: torch/tf (E2E) ────────────────────────────────────────
# golden 收框架原生张量：tf.* API 收 tf.EagerTensor，torch.* API 收 torch.Tensor
# 本算子无 aclnn API（CMakeLists ACLNNTYPE aclnn_exclude），故无 aclnn 通路 spec。


class TfFakeQuantWithMinMaxVarsSpec:
    """E2E (TF) 通路 — golden 收 tf.EagerTensor / torch.Tensor.

    e2e 通路（framework_api._call_plugin_with_plan）按框架 API 签名把标量
    属性连同张量一起按位置传入，故 num_bits/narrow_range 不得声明为
    keyword-only；kernel/geir 通路按 kwargs 传属性，两种传参都要兼容。
    golden 统一经 numpy 转成 torch 计算后返回 torch.Tensor，
    cpu_backend.result_to_numpy 对 torch/TF 张量均可用 .numpy() 回转。
    """

    def golden(x, min, max, num_bits=8, narrow_range=False, **kwargs):
        import torch

        x_t = torch.from_numpy(np.asarray(x))
        min_v = float(np.asarray(min).reshape(-1)[0])
        max_v = float(np.asarray(max).reshape(-1)[0])
        return [_forward(x_t, min_v, max_v, num_bits, narrow_range)]

    third_party = {
        "tf": _TfCompetitorE2E,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
