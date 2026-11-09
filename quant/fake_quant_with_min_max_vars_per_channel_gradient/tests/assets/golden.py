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

"""golden + third_party for fake_quant_with_min_max_vars_per_channel_gradient.

两条通路（kernel/geir + tf E2E），无 aclnn（CmakeLists aclnn_exclude）。
输出顺序：[backprops_wrt_x, backprop_wrt_min, backprop_wrt_max]。

golden 主实现为 torch 张量算子拼接（nudge 用 TF Nudge() 的 raw float32 公式，
仅 min==0 且 max==0 通道保留 TF 直通特例，其余退化通道走 raw 公式；
zp 舍入 round half away from zero；bpmin/bpmax 沿 leading axes 以 float64 累加）。
TF 降为 third_party 竞品（_TfCompetitor）做交叉核验。

环境要求：golden 主实现需要 torch（延迟导入）；third_party tf 竞品需要
tensorflow（延迟导入）。CI tf/torch 容器隔离，torch 为守卫式可选导入。
"""

import numpy as np

__spec__ = {
    # geir / kernel 通路：ttk kernel | GEIR，CSV op_name 查找，golden 收 numpy
    "fake_quant_with_min_max_vars_per_channel_gradient": "FakeQuantWithMinMaxVarsPerChannelGradientKernelSpec",
    # GE 类型名变体（CSV op_type / GE OpType，双注册，兼容两种 TTK 查找路径；ops-nn 同款键，并集只增不删）
    "FakeQuantWithMinMaxVarsPerChannelGradient": "FakeQuantWithMinMaxVarsPerChannelGradientKernelSpec",
    # E2E 通路：ttk e2e，CSV api_name = tf 竞品 API 全路径，golden 收 torch.Tensor
    "tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient": "TfFakeQuantWithMinMaxVarsPerChannelGradientSpec",
}

# TTK 外部插件注册：级别 -> {op_name -> 入口函数名}（ops-nn tests/assets 同款 legacy 约定，绝对 import only）
__golden__ = {
    "kernel": {
        "FakeQuantWithMinMaxVarsPerChannelGradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
        # Compatibility alias: some prepared TTK case CSVs bind op_name to the installed
        # opInterface value (snake_case) instead of the GE op_type (CamelCase).
        "fake_quant_with_min_max_vars_per_channel_gradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
    }
}


def _as_tensor(value):
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def fake_quant_with_min_max_vars_per_channel_gradient_golden(
    gradients, x, min, max, num_bits=8, narrow_range=False, **kwargs
):
    # ttk e2e calls golden with 6 positional args; attributes absent from the case
    # CSV arrive as the TF python-signature default None (the real op defaults live
    # in the op registration: num_bits=8, narrow_range=False). Accept positional or
    # keyword attrs and normalize None to the op defaults (issue #14).
    if num_bits is None:
        num_bits = 8
    if narrow_range is None:
        narrow_range = False
    import torch

    g = _as_tensor(gradients)
    x = _as_tensor(x)
    mn = _as_tensor(min).reshape(-1)
    mx = _as_tensor(max).reshape(-1)
    if x.dim() == 0:
        x = x.reshape(1)

    q_min = 1 if narrow_range else 0
    q_max = 2 ** int(num_bits) - 1
    channel = mn.shape[-1]
    q_range = float(q_max - q_min)

    # Nudge precision: float32, scale = (max - min) / qRange via a single Sub then a single
    # Div, matching the forward FakeQuantWithMinMaxVarsPerChannel and the TF reference
    # (README「约束说明」). Extreme/special min/max (+/-FLT_MAX, +/-inf) are outside the
    # documented domain; this reference keeps the plain fp32 evaluation for them.
    # No degeneracy short-circuit: TF's Nudge() applies the raw formula even when
    # max <= min (scale <= 0 reverses/collapses the nudged interval and the masks follow);
    # only min == 0 and max == 0 keeps the TF pass-through special case below.
    scale = (mx - mn) / q_range
    zp = q_min - mn / scale
    # Round half away from zero (TF std::round), not torch.round (round-half-to-even):
    # a zp_raw == n + 0.5 channel would otherwise shift the nudged interval by one scale.
    rounded = torch.sign(zp) * torch.floor(torch.abs(zp) + 0.5)
    nudged_zp = torch.clamp(rounded, float(q_min), float(q_max))
    nudged_min = (q_min - nudged_zp) * scale
    nudged_max = (q_max - nudged_zp) * scale

    shape = [1] * x.dim()
    shape[-1] = channel
    n_min = nudged_min.reshape(shape)
    n_max = nudged_max.reshape(shape)
    zero_both = ((mn == 0) & (mx == 0)).reshape(shape)

    between = ((x >= n_min) & (x <= n_max)).to(x.dtype)
    # Known benign divergence from the kernel's sentinel implementation for
    # zero_both channels: TF (and this reference) propagate bpx = g even when x
    # is NaN, while the kernel's [-inf, +inf] nudged-interval sentinel yields 0
    # for NaN comparisons (documented in the regbase kernel comment). Test data
    # does not cover NaN in this channel.
    bpx = torch.where(zero_both, g, g * between)

    min_mask = ((x < n_min) & ~zero_both).to(x.dtype)
    max_mask = ((x > n_max) & ~zero_both).to(x.dtype)
    # Both-zero channels contribute exactly zero (TF short-circuit): a plain g*mask would
    # give inf*0=NaN for +/-inf gradients and pollute the reductions.
    zero_g = torch.zeros_like(g)
    min_contrib = torch.where(zero_both, zero_g, g * min_mask)
    max_contrib = torch.where(zero_both, zero_g, g * max_mask)
    if x.dim() > 1:
        dims = tuple(range(x.dim() - 1))
        bpm = min_contrib.to(torch.float64).sum(dim=dims).to(x.dtype)
        bpmx = max_contrib.to(torch.float64).sum(dim=dims).to(x.dtype)
    else:
        bpm = min_contrib
        bpmx = max_contrib

    return [
        (bpx * 1.0).numpy(),
        bpm.numpy(),
        bpmx.numpy(),
    ]


class _TfCompetitor:
    def __init__(self, num_bits=None, narrow_range=None, **kwargs):
        self.num_bits = 8 if num_bits is None else int(num_bits)
        self.narrow_range = False if narrow_range is None else bool(narrow_range)

    def __call__(self, gradients, x, min, max, **kwargs):
        import tensorflow as tf

        try:
            import torch
        except ImportError:
            torch = None

        def arr(value):
            if torch is not None and isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        return tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(
            arr(gradients),
            arr(x),
            arr(min),
            arr(max),
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
        )


class FakeQuantWithMinMaxVarsPerChannelGradientKernelSpec:
    def golden(gradients, x, min, max, num_bits=8, narrow_range=False, **kwargs):
        # golden 主实现：torch 张量算子拼接（ops-nn 同步，与 kernel fp32 舍入链逐行对齐）；
        # num_bits/narrow_range 位置/关键字双兼容 + None 归一化（issue #14）。
        return fake_quant_with_min_max_vars_per_channel_gradient_golden(
            gradients, x, min, max, num_bits=num_bits, narrow_range=narrow_range
        )

    third_party = {"tf": _TfCompetitor}
    # 对齐算子仓修复定版口径（cann/ops-nn tests/assets/golden.py：mix_tolerance rtol=atol=1e-5）。
    # cross_check L1 比率判定在本 golden 形态下不可用：golden(torch 实现)与 third_party(TF) 为
    # 独立实现，部分元素 b-g 恰为 0，NPU 归约顺序的正常舍入差(4 ULP/rel≈4e-7)在该处无解
    # （L0_037 Output 1 实测）。golden 为真值，按 spec.yaml 容差判定。
    tolerance = {
        "float32": {"standard": "mix_tolerance", "rtol": 1.0e-5, "atol": 1.0e-5}
    }


class TfFakeQuantWithMinMaxVarsPerChannelGradientSpec:
    def golden(gradients, x, min, max, num_bits=8, narrow_range=False, **kwargs):
        return fake_quant_with_min_max_vars_per_channel_gradient_golden(
            gradients, x, min, max, num_bits=num_bits, narrow_range=narrow_range
        )

    third_party = {"tf": _TfCompetitor}
    # 同上：对齐算子仓修复定版口径（mix_tolerance rtol=atol=1e-5）
    tolerance = {
        "float32": {"standard": "mix_tolerance", "rtol": 1.0e-5, "atol": 1.0e-5}
    }
