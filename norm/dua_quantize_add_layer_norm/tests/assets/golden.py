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

"""TTK golden + third_party for dua_quantize_add_layer_norm.

单通路（geir/kernel）：本算子为 CANN 图融合产物（AddLayerNorm + 双 AscendQuant），
torch/tf 均无对应融合接口（无竞品），golden 用 torch 小算子拼接；无 aclnn API、
无 torch_npu/op-plugin 绑定，故无 aclnn / e2e 通路。

语义（归约轴固定为最后一维 W，与 kernel 计算链逐点对齐）：
  - x = (x1 + x2) + bias，逐元素先 Cast fp32 再累加；x 输出 CAST_RINT(RNE) cast 回输入 dtype。
  - 两遍法 LayerNorm：mu = Σx/W、var = Σ(x-mu)²/W（居中有偏）、rstd = 1/sqrt(var+eps)。
  - norm = ((x-mu)·rstd)·gamma + beta；scales/zero_points 为 [W] 参数沿 A 广播，zp 缺省按 0。
  - axis == -65535（乘法模式）：q = norm·scale + zp；其他任意值（除法模式）：q = norm/scale + zp。
    axis 只是缩放方式选型开关，不是归约轴下标。zp 在 rint 之前加入；
    y = int8(clip(rint(q), -128, 127))，NaN→0、±Inf→±127/-128（SAT=clip）。

精度策略：
  - int8 输出(y1/y2) 配 binary_equal（整型输出逐位一致：量化链 Div/Sqrt 0ULP + CAST_RINT
    取偶与 golden IEEE 口径对齐，不设 ±1 LSB 放宽）；
    浮点输出(x) 配 cross_check L0（小算子拼接算子统一 L0 档：mare≤10/mere≤2/rmse≤2）。
  - arch22(A2) 实现偏差（norm_out 先 CAST_RINT 到输入 dtype 再回 fp32、mean/var 走
    Σ(x·1/W) 逐元素乘、A2 二进制仅 bf16 tilingKey 1000/1001）单项 ≤1 LSB，**当前实现中
    仍存在、未修复**；本 golden 不再用容差吸收，arch22 上 int8 可能因此判红，需按实现
    缺陷修复（arch35 已实测逐位一致）。
  - dtype：float16 / bfloat16 / float32（bf16 经 ml_dtypes 按位 view 桥接）。
"""

import numpy as np
import torch

__spec__ = {
    # geir / kernel 通路：ttk kernel | GEIR，CSV op_name 查找，golden 收 numpy
    "dua_quantize_add_layer_norm": "DuaQuantizeAddLayerNormKernelSpec",
}


# ─── dtype 桥与输入规范化（bf16 按位 view / rank-0 按 {1} / 可选缺省哨兵）──


def _to_torch(arr):
    """numpy(含 ml_dtypes.bfloat16) -> torch tensor(原 dtype,按位 view 无损机械转换)。"""
    arr = np.asarray(
        arr
    )  # numpy 标量 -> 0-d ndarray（ascontiguousarray 会把 0-d 升成 1-d）
    name = arr.dtype.name
    if name == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).view(np.int16)).view(torch.bfloat16)
    if arr.ndim == 0:
        return torch.from_numpy(arr)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _from_torch(t, torch_dtype):
    """torch tensor -> numpy(含 ml_dtypes.bfloat16)，按位 view 无损；dtype 用 torch dtype 判型。"""
    if torch_dtype == torch.bfloat16:
        import ml_dtypes

        return t.view(torch.int16).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
    return t.numpy()


def _row(t):
    """rank-0 标量按 {1} 处理（host EnsureNotScalar 同口径）。"""
    return t.reshape(1) if t.dim() == 0 else t


def _opt(t):
    """可选输入哨兵：None 或空 tensor 视为缺省（host 判据 shape+desc 双非空，缺省 zp=0）。"""
    return None if t is None or t.numel() == 0 else t


# ─── 前向计算（kernel 逐点对齐：arch35 VF1/VF6/VF8/VF10 + VF11~VF14，arch22 同构）──

# axis 仅是量化缩放方式的选型开关（-65535 -> 乘法，其他任意值 -> 除法），
# 不是归约轴下标。host 侧同判据：arch35 MUL_MODE_AXIS / arch22 AXIS_VALUE_FOR_MUL_MODE。
_MUL_MODE_AXIS = -65535


def _quantize(norm, scale, zero_point, is_mul):
    """q = rint(norm(*|/)scale + zp) -> NaN->0 -> clip(-128,127) -> int8。

    顺序对齐内核 VF11/VF13→VF12/VF14：zp 在 rint 之前加入；round-half-to-even
    对齐 CAST_RINT；NaN 先归 0 再饱和（±Inf 不动，由 clip 饱和到 ±127/-128）。
    """
    s = scale.to(torch.float32)
    q = norm * s if is_mul else norm / s
    if zero_point is not None:
        q = q + zero_point.to(torch.float32)
    q = torch.round(q)
    q = torch.nan_to_num(q, nan=0.0)
    return torch.clamp(q, -128, 127).to(torch.int8)


def _forward(
    x1,
    x2,
    gamma,
    beta,
    bias,
    scales1,
    scales2,
    zero_points1=None,
    zero_points2=None,
    *,
    axis=-1,
    epsilon=1e-5,
):
    """前向：fp32 域 Add -> 两遍法 LayerNorm -> 双路量化，返回 [y1, y2, x]（顺序照 def.cpp）。

    全程 fp32 累加是内核权威语义（VF1/arch22 逐元素 Cast fp32 再 Add，非精度决策）；
    x 输出无条件写出（additional_output 属性实现不消费）。
    """
    try:
        axis = int(axis)
    except (TypeError, ValueError):
        axis = -1
    epsilon = float(epsilon)
    is_mul = axis == _MUL_MODE_AXIS

    orig_shape = x1.shape
    x1 = _row(x1)
    x2 = _row(x2)
    if x1.shape[-1] <= 0:
        # 域外输入：host CheckSupported 对 W<=0 硬拒收（W=0 到不了 empty 分支）。
        raise ValueError("x1 last dim (W) must be greater than 0")

    # (x1 + x2) + bias，逐元素 fp32；gamma/beta/bias/scales/zp 为 [W] 参数沿 A 广播。
    xf = x1.to(torch.float32) + x2.to(torch.float32) + _row(bias).to(torch.float32)
    x_out = xf.to(x1.dtype)  # CAST_RINT = RNE，torch .to 同口径

    mu = xf.mean(dim=-1, keepdim=True)  # Σ/W（arch35 VF6 0ULP Div 同口径）
    xc = xf - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)  # 居中/有偏 var，Σ/W
    rstd = 1.0 / torch.sqrt(var + epsilon)  # Div(1, Sqrt) 两步，非 rsqrt（VF8 同口径）

    norm = xc * rstd * _row(gamma).to(torch.float32) + _row(beta).to(torch.float32)
    y1 = _quantize(norm, _row(scales1), _opt(zero_points1), is_mul)
    y2 = _quantize(norm, _row(scales2), _opt(zero_points2), is_mul)
    return [y1.view(orig_shape), y2.view(orig_shape), x_out.view(orig_shape)]


# ─── third_party torch 标杆（class form，无竞品 -> torch 拼接）──────────


class _TorchCompetitor:
    """third_party torch — 同一 torch 拼接在远端 GPU 上执行。

    竞品最优形态（性能对标铁律）：torch.compile 融合（eager 逐算子 launch 全是开销，
    NPU 侧是单个融合内核），编译失败回落 eager。__init__ 收属性（axis/epsilon），
    __call__ 收 def.cpp 同名输入 tensor；x 输出 cast 回输入 dtype（竞品留 fp32 会让
    cross_check 的 ratio 凭空爆表）。
    """

    def __init__(self, *, axis=-1, epsilon=1e-5, **kwargs):
        try:
            self.axis = int(axis)
        except (TypeError, ValueError):
            self.axis = -1
        self.epsilon = float(epsilon)
        self._compiled = None

    def _impl(
        self,
        x1,
        x2,
        gamma,
        beta,
        bias,
        scales1,
        scales2,
        zero_points1=None,
        zero_points2=None,
    ):
        xf = x1.to(torch.float32) + x2.to(torch.float32) + bias.to(torch.float32)
        x_out = xf.to(x1.dtype)
        mu = xf.mean(dim=-1, keepdim=True)
        xc = xf - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + self.epsilon)
        norm = xc * rstd * gamma.to(torch.float32) + beta.to(torch.float32)
        is_mul = self.axis == _MUL_MODE_AXIS
        y1 = _quantize(norm, scales1, zero_points1, is_mul)
        y2 = _quantize(norm, scales2, zero_points2, is_mul)
        return [y1, y2, x_out]

    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta,
        bias,
        scales1,
        scales2,
        zero_points1=None,
        zero_points2=None,
        **kwargs,
    ):
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(
                x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2
            )
        except Exception:
            # 永久降级到 eager：避免 compiled 每次调用重复抛异常（逐用例开销且污染 perf 计时）
            self._compiled = self._impl
            return self._impl(
                x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2
            )


# ─── 通路 1: geir / kernel (GEIR) ─────────────────────────────────
# golden 收 numpy.ndarray，third_party 收 torch.Tensor。
# 无 aclnn API、无 torch_npu/op-plugin 绑定，故无 aclnn / e2e 通路 spec
# （同 multi_add_rms_norm_dynamic_quant 先例）。


class DuaQuantizeAddLayerNormKernelSpec:
    """geir / kernel 通路 — golden 收 numpy.ndarray."""

    def golden(
        x1,
        x2,
        gamma,
        beta,
        bias,
        scales1,
        scales2,
        zero_points1=None,
        zero_points2=None,
        *,
        axis=-1,
        epsilon=1e-5,
        **kwargs,
    ):
        x1_t = _to_torch(x1)
        y1, y2, x = _forward(
            x1_t,
            _to_torch(x2),
            _to_torch(gamma),
            _to_torch(beta),
            _to_torch(bias),
            _to_torch(scales1),
            _to_torch(scales2),
            _to_torch(zero_points1) if zero_points1 is not None else None,
            _to_torch(zero_points2) if zero_points2 is not None else None,
            axis=axis,
            epsilon=epsilon,
        )
        return [y1.numpy(), y2.numpy(), _from_torch(x, x1_t.dtype)]

    third_party = {
        "torch": _TorchCompetitor,
    }
    tolerance = {
        "int8": {"standard": "binary_equal"},
        "float16": {"standard": "cross_check", "level": "L0"},
        "bfloat16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }
