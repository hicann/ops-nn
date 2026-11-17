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
"""Three-alpha Celu golden and third-party reference (Torch).

数学合同 (与算子注册面一致):
    x >= 0 : y = alpha3 * x
    x <  0 : y = alpha1 * (exp(x / alpha2) - 1)

计算核以 torch.nn.functional.celu 为基准组合: 对输入按 alpha2 调用标准
单 alpha CELU 得 u (x >= 0 时 u = x, x < 0 时 u = alpha2 * expm1(x / alpha2)),
再按分支缩放 —— 正支 alpha3 * u, 负支 (alpha1 / alpha2) * u —— 得到三
alpha 结果; alpha1 = alpha2 = alpha 且 alpha3 = 1 时与
torch.nn.functional.celu(x, alpha) 逐点一致。alpha2 非有限 (如 +Inf) 或
alpha2 / 负支配比超出计算精度可表示范围时, 组合路径中间量会失真,
退回数学合同公式的 Torch 原语直算 (负支 alpha1 * expm1(min(x / alpha2,
87.0)), 87.0 为公式的 exp 输入数值保护上界, expm1 单次舍入消近零区
两次舍入抵消; IEEE 754 语义: x / alpha2 = -0.0 时 expm1(-0.0) = -0.0,
alpha2 = +Inf 时负支随商趋 -0.0 而饱和到 0)。

通路与载体:
    - Kernel / GEIR: golden 收 numpy.ndarray, 转 torch 计算后以 numpy 返回;
    - third_party: torch.Tensor 直调 _TorchCelu 核心 (与 golden 同一
      精度合同)。

精度契约 (纯函数自保证, 不依赖测试框架升精度):
    golden 不读任何控制形参 —— 测试框架传入的 golden_mode 等控制参数
    一律经 **kwargs 吸收, 对计算精度与分支零影响。fp16 到达先无损升
    fp32 计算, 结果经 RNE (最近偶舍入) 回投 fp16 返回 —— 镜像 kernel
    入口 CAST_NONE 无损升 / 出口 CAST_ROUND 回投的精度合同; fp32/fp64
    到达按到达精度直算直返。不依赖测试框架是否升精度 ("Promote" 形态
    下 fp16 被框架升 fp32 后到达, 与本合同的 fp32 分支自然衔接)。
    合同在 _TorchCelu 计算核内承载, third_party 直调同一核心, 与
    golden 双角色统一同合同 (三方同样不依赖框架 Promote)。kernel
    输出的量化残差由输出 dtype 键控的容差吸收。

torch 仅在函数体内加载: 本文件可能在已初始化设备运行时的进程内被
加载, 模块级框架加载有进程内冲突风险; math 可全局加载。
"""

import math

__spec__ = {
    "celu": "CeluTestSpec",
    "Celu": "CeluTestSpec",
}

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _normalize_alphas(alpha1, alpha2, alpha3):
    import torch

    alpha1 = torch.tensor(alpha1, dtype=torch.float32).item()
    alpha2 = torch.tensor(alpha2, dtype=torch.float32).item()
    alpha3 = torch.tensor(alpha3, dtype=torch.float32).item()
    if not alpha2 > 0.0:
        raise ValueError("alpha2 must be greater than zero")
    return alpha1, alpha2, alpha3


def _to_torch(x):
    import torch

    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _return_like_input(result, reference):
    import torch

    if isinstance(reference, torch.Tensor):
        return result.to(reference.device)
    return result.detach().cpu().numpy()


class _TorchCelu:
    """Three-alpha Celu composed from the single-alpha
    torch.nn.functional.celu.

    双重角色: golden 计算核与三方参考实现。双角色统一精度合同: fp16
    到达升 fp32 计算 + RNE 回投 fp16, fp32/fp64 直算直返 (合同在本类
    __call__ 核心承载, 三方与 golden 同源同合同)。
    """

    def __init__(self, alpha1=1.0, alpha2=1.0, alpha3=1.0, **kwargs):
        self.alpha1, self.alpha2, self.alpha3 = _normalize_alphas(
            alpha1, alpha2, alpha3
        )

    def __call__(self, x, **kwargs):
        import torch

        if x.dtype not in {torch.float16, torch.float32, torch.float64}:
            raise TypeError(f"unsupported Celu dtype: {x.dtype}")
        # fp16 到达: 无损升 fp32 计算, 结果 RNE (最近偶舍入) 回投 fp16
        # —— 镜像 kernel 入口 CAST_NONE 无损升 / 出口 CAST_ROUND 回投;
        # 纯函数自保证, 不依赖框架升精度, golden 与 third_party 双角色
        # 统一精度合同。
        is_fp16 = x.dtype == torch.float16
        if is_fp16:
            x = x.to(torch.float32)
        # 组合路径要求 alpha2 与负支配比 alpha1 / alpha2 都在计算精度
        # (fp16 到达已升 fp32, finfo/分支选择与 kernel 的 fp32 量级
        # 分类镜像一致) 内可表示: alpha2 过小时标准 API 的 alpha 标量
        # 换算会失真或报错, 支配比过大时分支缩放会溢出; 不可表示时退
        # 回原语逐算子组合。
        finfo = torch.finfo(x.dtype)
        branchScale = self.alpha1 / self.alpha2
        if (
            math.isfinite(self.alpha2)
            and self.alpha2 >= finfo.tiny
            and math.isfinite(branchScale)
            and abs(branchScale) <= finfo.max
        ):
            standard = torch.nn.functional.celu(x, alpha=self.alpha2)
            result = torch.where(
                x >= 0,
                torch.mul(standard, self.alpha3),
                torch.mul(standard, branchScale),
            )
        else:
            # alpha2 非有限或超出计算精度可表示范围: 退回原语逐算子
            # 组合。87.0 为 exp 输入的数值保护上界
            # (exp(87.0) < fp32 max); 负支域内 (x < 0 且 alpha2 > 0)
            # x / alpha2 恒为负, 该截断不改变结果, 仅防御异常输入。
            # expm1 单次舍入: exp(ratio)-1 两次舍入在 ratio -> 0^- 时
            # 灾难性抵消 (exp 舍入到 1.0 差值为 0, 乘 alpha1 后错值);
            # expm1(-0.0) = -0.0 保持零符号语义。
            ratio = torch.clamp(torch.div(x, self.alpha2), max=87.0)
            negative = torch.mul(self.alpha1, torch.expm1(ratio))
            positive = torch.mul(self.alpha3, x)
            result = torch.where(x >= 0, positive, negative)
        if is_fp16:
            result = result.to(torch.float16)
        return [result]


def _celu_golden(
    x,
    alpha1=1.0,
    alpha2=1.0,
    alpha3=1.0,
    **kwargs,
):
    # fp16 升 fp32 计算 + RNE 回投的合同已在 _TorchCelu 核心承载
    # (双角色统一精度合同), 本入口直通。
    source = _to_torch(x)
    result = _TorchCelu(alpha1, alpha2, alpha3)(source)[0]
    return [_return_like_input(result, x)]


class CeluTestSpec:
    """Three-alpha Celu test specification (Kernel / GEIR pathways)."""

    golden = staticmethod(_celu_golden)
    third_party = {"torch": _TorchCelu}
    tolerance = _TOLERANCE
