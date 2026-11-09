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

"""
Golden function for FakeQuantWithMinMaxVarsPerChannel operator.

同时作为 TTK kernel/geir/e2e 模式的外部 golden 插件使用：
    python3 -m ttk kernel -i xxx.csv --plugin golden.py
TTK 靠 AST 静态解析顶层 __golden__ 字典，并以
golden_func(*context.input_arrays, **kwargs) 方式调用入口函数
（num_bits/narrow_range 等属性经 kwargs 传入；geir 通路与 kernel 共用
kernel 级注册）。kernel 级同时注册 GE 类型名与设计接口名（snake_case），
覆盖准备层两种 op_name 写法。

实现与算子 kernel（op_kernel/arch35 v2.2 方案 A）计算链逐行对齐：

  Nudge:  scale = (max - min) / (qMax - qMin)   [fp32 单步除法]
          inv_scale = (qMax - qMin) / (max - min)
          zp = qMin - min / scale（除法，非乘 inv_scale）
          nudged_zp = clamp(round_half_away(zp), qMin, qMax)
  退化:   0 <= scale < FLT_MIN_NORMAL(≈1.17549435e-38) 即退化通道，输出全零
          （min==max 时 scale 必为 0；次正规区 0<scale<FLT_MIN 存在设备侧 FTZ
          与参考侧次正规运算分歧及倒数上溢，统一并入退化——与 kernel keep
          判定一致）。负 scale（min>max 反向区间，README 不支持域）不判退化。
  主公式: out = floor((clamped - nudgedMin) * inv_scale + 0.5) * scale + nudgedMin
  clamp:  clamped = max(min(x, nudgedMax), nudgedMin) —— 先 Min 后 Max，
          与 kernel 的 Min/Max 指令链及 TF 竞品一致：min>max 的反向通道
          （README 列为不支持）三条实现一致回退到 nudgedMin，避免域外
          输入产生虚假精度分歧（torch.clamp 反向时取 max_bound 会与
          kernel/TF 相反）。
  NaN:    x 为 NaN 时经 min/max 传播，y 为 NaN（README 功能说明）；
          退化通道的 NaN 车道同样保持 NaN（NaN*0 = NaN）。
  精度:   统一 float32 计算（含 TTK Promote 升精度输入）——golden 镜像 kernel
          的 fp32 舍入链，升 fp64 会在 zp 半整数边界翻转 1 个量化级。

竞品与精度策略（沿用 operator-artifacts Golden 文件口径）：
  - 唯一真实竞品为 TensorFlow tf.quantization.fake_quant_with_min_max_vars_per_channel
    （REQUIREMENTS §3 / DESIGN §10.1）；本算子 CMakeLists 声明 ACLNNTYPE
    aclnn_exclude，无 aclnn 通路，kernel/geir 与 TF 为仅有的可比对实现。
  - DESIGN §6 记录精度分歧：元素级量化用 round-half-up（floor(x+0.5)），
    TF 用 round-half-away-from-zero，负数 0.5 边界处 1 个量化级别偏移；
    故 golden 与 kernel 计算链逐行对齐，TF 仅作三方比对参考，不作精度真值。
  - 比对标准建议 mix_tolerance（golden 为真值）。cross_check 需三方端点，
    无 GPU/XPU 环境时逐例 GOLDEN_FAILURE，属环境限制。

dtype 支持 float32（README 支持组合）；Promote 输入降回 fp32 计算以镜像 kernel。
"""

# Standard Packages
from typing import Tuple

# Third-Party Packages
import numpy as np
import torch

# TTK 外部插件注册：级别 -> {op_name -> 入口函数名}（绝对 import only，勿加相对 import）
# kernel 级双注册：GE 类型名（投影后 CSV）+ 设计接口名（Host 准备的隔离用例）
__golden__ = {
    "kernel": {
        "FakeQuantWithMinMaxVarsPerChannel": "fake_quant_with_min_max_vars_per_channel_golden",
        "fake_quant_with_min_max_vars_per_channel": "fake_quant_with_min_max_vars_per_channel_golden",
    },
    "e2e": {
        "tf.quantization.fake_quant_with_min_max_vars_per_channel": "fake_quant_with_min_max_vars_per_channel_e2e_golden",
    },
}

# ─── __spec__ TestSpec 显式注册（与 opencann/operator-artifacts 保持一致）─────────
__spec__ = {
    "fake_quant_with_min_max_vars_per_channel": "FakeQuantWithMinMaxVarsPerChannelKernelSpec",
    "FakeQuantWithMinMaxVarsPerChannel": "FakeQuantWithMinMaxVarsPerChannelKernelSpec",
    "tf.quantization.fake_quant_with_min_max_vars_per_channel": "TfFakeQuantWithMinMaxVarsPerChannelSpec",
}

# fp32 最小正常数：次正规区（0 < |值| < FLT_MIN）按退化处理（与 kernel keep 判定一致）
_FLT_MIN_NORMAL = 1.1754943508222875e-38
# fp32 最大有限数：量程 (max-min) 超过它即溢出为 +inf，启用防溢出分解（与 kernel 2.1b 一致）
_FLT_MAX = 3.4028234663852886e38
# 倒数次正规判定界（P1-01）：scale > 1/FLT_MIN ⟺ 对应倒数落入次正规区（与 kernel 一致）
_INV_SUB_SCALE_BOUND = 8.507059173023462e37


def _as_float_tensor(value):
    """统一按 float32 计算。

    golden 的定位是 kernel fp32 舍入链的逐行镜像（DESIGN §6）；TTK 在
    mix_tolerance 下会以 Promote 升精度（float64 输入）调用 golden，此处
    必须降回 fp32——fp64 下 zp 半整数边界（如 127.5）的舍入方向与 fp32
    kernel 相差 1 个量化级，保留高精度反而引入虚假精度分歧。
    """
    return torch.as_tensor(value, dtype=torch.float32)


def _nudge_per_channel(min_vec, max_vec, num_bits, narrow_range):
    """Per-channel Nudge 预计算（与 kernel 阶段 2 对齐）。

    返回 (nudged_min, nudged_max, scale, inv_scale, keep)，均 shape (C,)；
    keep = 1 为正常通道，0 为退化通道（输出清零）。
    """
    q_min = 1.0 if narrow_range else 0.0
    q_max = float(2**num_bits - 1)
    q_range = q_max - q_min
    # 量程溢出判定（P2-01 统一：全链仅此一种判据形式，不经过 (max-min) 溢出点）：
    # max > FLT_MAX + min（min<0 时 = FLT_MAX-|min|，max>它 ⟺ max-min>FLT_MAX）
    overflow = max_vec > (_FLT_MAX + min_vec)
    q_range_t0 = torch.tensor(q_range, dtype=min_vec.dtype)
    scale = torch.where(
        overflow,
        max_vec / q_range_t0 - min_vec / q_range_t0,
        (max_vec - min_vec) / q_range_t0,
    )
    # 退化检测（P1-01 补充：双侧次正规）：|scale| < FLT_MIN_NORMAL 即退化。
    # 正侧涵盖 min==max（scale==0）与正次正规带；负侧涵盖反向区间×次正规量程
    # （min>max 且 |max-min| 次正规：倒数溢出 ∓inf、设备 FTZ 清零，反向回退链
    # 的 int32 转换越界 UB——统一按退化清零，kernel 同步）
    degenerate = (scale < _FLT_MIN_NORMAL) & (scale > -_FLT_MIN_NORMAL)
    keep = (~degenerate).float().to(min_vec.dtype)
    # K1(Issue #101)：min==max!=0 相等通道输出 NaN（对齐 TF oracle）
    eq_nan = (min_vec == max_vec) & (min_vec != 0)
    # 避免除零/NaN：退化通道的 min/max 置安全值（结果会被 keep 清零）
    safe_min = torch.where(degenerate, torch.zeros_like(min_vec), min_vec)
    safe_max = torch.where(degenerate, torch.ones_like(max_vec), max_vec)
    # safe 值上以同一判据形式重判（退化通道替换后量程有限，仅正常溢出通道命中）
    ovf2 = safe_max > (_FLT_MAX + safe_min)
    # 注意：q_range 是 python float，q_range/tensor 会先提升到 fp64 再舍入（与设备
    # fp32 单步除法差 1 ULP，在整数边界翻转 floor 一级）——统一改为 fp32 tensor 除法
    q_range_t = torch.tensor(q_range, dtype=min_vec.dtype)
    scale = torch.where(
        ovf2,
        safe_max / q_range_t - safe_min / q_range_t,
        (safe_max - safe_min) / q_range_t,
    )
    inv_scale = torch.where(
        ovf2,
        torch.tensor(1.0, dtype=min_vec.dtype) / scale,
        q_range_t / (safe_max - safe_min),
    )
    zp = q_min - safe_min / scale
    # round-half-away then clamp（kernel 阶段 2.5：clamp(round_half_away(zp), qMin, qMax)）
    rounded_zp = torch.sign(zp) * torch.floor(torch.abs(zp) + 0.5)
    nudged_zp = torch.clamp(rounded_zp, q_min, q_max)
    nudged_min = (q_min - nudged_zp) * scale
    nudged_max = (q_max - nudged_zp) * scale
    # P1-02：仅当计算值越过 ±FLT_MAX（∓inf）时按原 min/max 收边；有限值保持原值
    # （zp 舍入可使有限 nudgedMin 低于 min 达 0.5 步长，无条件收边会平移量化网格）
    nudged_min = torch.where(nudged_min < -_FLT_MAX, safe_min, nudged_min)
    nudged_max = torch.where(nudged_max > _FLT_MAX, safe_max, nudged_max)
    # P1-01：scale > 1/FLT_MIN ⟺ 对应倒数落入次正规区（设备 FTZ 清零），此类通道
    # 改用除法先行并纳入双侧重建（与 kernel maskInvSub/maskRecon 一致）
    inv_sub = scale > _INV_SUB_SCALE_BOUND
    # P1-03：双侧重建分界取索引区间半程 qRange/2
    half = (q_max - q_min) / 2.0
    recon = overflow | inv_sub
    return nudged_min, nudged_max, scale, inv_scale, keep, overflow, recon, half, eq_nan


def _forward(x_t, min_t, max_t, num_bits, narrow_range):
    """公共前向：clamp 链与 kernel Min/Max 指令及 TF 一致（先 Min 后 Max）。"""
    C = min_t.shape[-1]
    nudged_min, nudged_max, scale, inv_scale, keep, ovf, recon, half, eq_nan = (
        _nudge_per_channel(min_t, max_t, num_bits, narrow_range)
    )

    # 将 (C,) 广播到 x 的 shape（最后一维对齐 channel）
    broadcast_shape = [1] * (x_t.dim())
    broadcast_shape[-1] = C
    n_min = nudged_min.reshape(broadcast_shape)
    n_max = nudged_max.reshape(broadcast_shape)
    sc = scale.reshape(broadcast_shape)
    inv_sc = inv_scale.reshape(broadcast_shape)
    kp = keep.reshape(broadcast_shape)
    ob = ovf.reshape(broadcast_shape)
    rb = recon.reshape(broadcast_shape)

    # clamp：先 min(x, nudgedMax) 再 max(·, nudgedMin)，与 kernel Min/Max 链及 TF 一致
    # （torch.clamp 在 min>max 反向通道取 max_bound，与 kernel/TF 相反，故不用 clamp）
    clamped = torch.maximum(torch.minimum(x_t, n_max), n_min)
    # round-half-up: floor((clamped - n_min) * inv_scale + 0.5)
    shift = clamped - n_min
    scaled = shift * inv_sc + 0.5
    # CODE.11 ③溢出通道 scaled 分解（乘 invScale 形式，与 kernel 阶段 3 一致）
    scaled_alt = clamped * inv_sc - n_min * inv_sc + 0.5
    scaled = torch.where(ob, scaled_alt, scaled)
    # P1-01 倒数次正规通道：scaled 改用除法先行（scale 为正常数，无次正规参与，
    # 覆盖设备 FTZ 清零倒数的情形；对溢出通道的次正规 1/scaleAlt 同样生效）
    inv_sub_b = sc > _INV_SUB_SCALE_BOUND
    scaled_div = clamped / sc - n_min / sc + 0.5
    scaled = torch.where(inv_sub_b, scaled_div, scaled)
    floored = torch.floor(scaled)
    out = floored * sc + n_min
    # CODE.11 ④双侧重建（P1-03 分界 qRange/2；适用通道=溢出∪倒数次正规）：
    # i<=half 用 nMin+i*sc、i>half 用 nMax-(qRange-i)*sc（P1-02：高侧距离用
    # qRange-floored，narrow 时 qMax-floored 会多 qMin 步）
    q_max = float(2**num_bits - 1)
    q_min_v = 1.0 if narrow_range else 0.0
    q_range_v = q_max - q_min_v
    out_lo = n_min + floored * sc
    out_hi = n_max - (q_range_v - floored) * sc
    out_alt = torch.where(floored > half, out_hi, out_lo)
    out = torch.where(rb, out_alt, out)
    # P1-06 兜底 1：|out|>FLT_MAX（±inf）→ 取另一分支；兜底 2：仍越界（对称中点
    # 角落两分支同时越界）→ 取 clamped（与真值差 ≤1 个量化步长，好于 ±inf）
    other = torch.where(floored > half, out_lo, out_hi)
    bad = (out - _FLT_MAX > 0) | (out + _FLT_MAX < 0)
    out = torch.where(bad, other, out)
    bad = (out - _FLT_MAX > 0) | (out + _FLT_MAX < 0)
    out = torch.where(bad, clamped, out)
    # 退化兜底：keep=0 通道清零（NaN 车道 NaN*0=NaN，保持 NaN 透传语义）
    out = out * kp
    # K1(Issue #101)：min==max!=0 相等通道注入 NaN（对齐 TF oracle），
    # min==max==0 全零通道保持上方清零结果
    en_b = eq_nan.reshape(broadcast_shape)
    out = torch.where(en_b, torch.full_like(out, float("nan")), out)
    return out


def fake_quant_with_min_max_vars_per_channel_golden(
    x: np.ndarray,
    min: np.ndarray,
    max: np.ndarray,
    num_bits: int = 8,
    narrow_range: bool = False,
    **kwargs,
) -> Tuple[np.ndarray]:
    """kernel/geir 通路 golden（输入为 numpy 数组，属性经 kwargs 传入）。

    Args:
        x: 输入 Tensor，最后一轴为 channel C
        min: per-channel 量化下界，1-D（长度 C）
        max: per-channel 量化上界，1-D（长度 C）
        num_bits: 量化位宽，[2, 16]，默认 8
        narrow_range: 窄量化范围（qMin=1），默认 False

    Returns:
        Tuple[np.ndarray]: 单输出 y（shape 与 x 一致）
    """
    num_bits = int(num_bits)
    narrow_range = bool(narrow_range)
    x_t = _as_float_tensor(x)
    min_t = _as_float_tensor(min).reshape(-1)
    max_t = _as_float_tensor(max).reshape(-1)
    out = _forward(x_t, min_t, max_t, num_bits, narrow_range)
    return (out.numpy(),)


def fake_quant_with_min_max_vars_per_channel_e2e_golden(
    x, min, max, num_bits: int = 8, narrow_range: bool = False, **kwargs
) -> Tuple[np.ndarray]:
    """e2e（TF）通路 golden：输入为 torch.Tensor / numpy 均可，按输入精度计算。

    竞品为 tf.quantization.fake_quant_with_min_max_vars_per_channel；本 golden
    与 kernel 计算链对齐（round-half-up），TF 仅作三方比对参考（DESIGN §6）。
    """
    num_bits = int(num_bits)
    narrow_range = bool(narrow_range)
    x_t = _as_float_tensor(x)
    min_t = _as_float_tensor(min).reshape(-1)
    max_t = _as_float_tensor(max).reshape(-1)
    out = _forward(x_t, min_t, max_t, num_bits, narrow_range)
    return (out.numpy(),)


# ─── third_party TF 竞品（与 opencann/operator-artifacts 保持一致）───────────────
# 唯一竞品为 TensorFlow tf.quantization.fake_quant_with_min_max_vars_per_channel
# （REQUIREMENTS §3 / DESIGN §10.1）。TF per-channel API 的 min/max 即为
# 1-D tensor，与本算子接口一致，直接调用。server 侧 import tensorflow（standalone）。


class _TfCompetitor:
    """third_party tf — 调用 tf.quantization.fake_quant_with_min_max_vars_per_channel."""

    def __init__(self, *, num_bits=8, narrow_range=False, **kwargs):
        self.num_bits = int(num_bits)
        self.narrow_range = bool(narrow_range)

    def __call__(self, x, min, max, **kwargs):
        import tensorflow as tf

        return [
            tf.quantization.fake_quant_with_min_max_vars_per_channel(
                x, min, max, num_bits=self.num_bits, narrow_range=self.narrow_range
            )
        ]


# ─── TestSpec 类（__spec__ 声明的目标）─────────────────────────────────────────


class FakeQuantWithMinMaxVarsPerChannelKernelSpec:
    """geir / kernel 通路 — golden 收 numpy.ndarray."""

    def golden(x, min, max, *, num_bits=8, narrow_range=False, **kwargs):
        return fake_quant_with_min_max_vars_per_channel_golden(
            x, min, max, num_bits, narrow_range
        )

    third_party = {
        "tf": _TfCompetitor,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }


class TfFakeQuantWithMinMaxVarsPerChannelSpec:
    """E2E (TF) 通路 — golden 收 torch.Tensor."""

    def golden(x, min, max, *, num_bits=8, narrow_range=False, **kwargs):
        return fake_quant_with_min_max_vars_per_channel_e2e_golden(
            x, min, max, num_bits, narrow_range
        )

    third_party = {
        "tf": _TfCompetitor,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
