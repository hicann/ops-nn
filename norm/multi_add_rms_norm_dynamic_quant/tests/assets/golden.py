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
TTK TestSpec for multi_add_rms_norm_dynamic_quant (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（add/mean/rsqrt/round/clamp/amax），
                   全程 fp32 累加（A2 权威语义，非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一 torch 拼接在远端 GPU 上执行（torch.compile 融合，竞品最优形态）
    tolerance    —— int8 输出(y1/y2) 配 quant（绝对误差 ≤1 LSB；.5 边界舍入方向属实现
                   自由度，逐位相等=要求复现浮点运算次序，会大面积误判）；浮点输出
                   (x/y 为 fp16/bf16，scale1/scale2 为 fp32) 配 cross_check

Canonical IO order (multi_add_rms_norm_dynamic_quant_def.cpp):
    inputs : x1(TensorList, DYNAMIC), x2, gamma, smooth_scale1(OPT), smooth_scale2(OPT)
    outputs: y1(int8), y2(int8), x(同 x1 dtype), y(同 x1 dtype), scale1(fp32), scale2(fp32)
    attrs  : epsilon(OPTIONAL float=1e-6)

语义:
    x = (x1[0]+x2) + x1[1] + ...     (全程 fp32 累加,A2 顺序:x2 第二个加;x 输出 cast 回 dtype)
    rstd = 1/sqrt(mean(x^2)+eps)     (fp32,末轴)
    normed = x * rstd * gamma        (fp32) => y 输出(cast dtype)
    对每个 smooth_scale_j:
      q = normed * smooth_j
      scale_j = max(|q|)/127         (逐行,fp32)
      y_j = round_half_even(q/scale_j) -> clamp(-128,127) -> int8
    无 smooth 时 q = normed。
y2/scale2 有效 <=> smooth_scale2 存在(smoothNum==2)。对齐 A2/arch35 内核(hasY2Scale2_=hasSmoothScale2_,
smoothNum<2 时内核不写 y2/scale2、留 buf 初值)——golden 对它们返回 None 哨兵,TTK 比对跳过
(core_modules/npu/op/comparison.py: None/str 哨兵 continue),避免拿内核未写的垃圾值误判 FAIL。
"""

import numpy as np
import torch

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
# int8 量化输出必须显式配 quant——不声明则整数短路 binary_equal，round 在 .5 边界的
# ±1 LSB tie 会大面积误判红（gnsq 实证）。
_TOL = {
    "int8": {"standard": "quant"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _to_torch(arr):
    """numpy(含 ml_dtypes.bfloat16) -> torch tensor(原 dtype,按位 view 无损机械转换)。"""
    name = arr.dtype.name
    if name == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _from_torch(t, np_dtype_name):
    if np_dtype_name == "bfloat16":
        import ml_dtypes

        return t.view(torch.int16).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
    return t.numpy()


def _compute(x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs):
    """torch.Tensor 进 / 出（x1 为 list[Tensor]），返回 [y1, y2, x, y, scale1, scale2]，
    顺序照 def.cpp；y2/scale2 在 smooth_scale2 缺省时为 None 哨兵。
    全程 fp32 累加是 A2 权威语义（算子定义如此，非精度决策）。"""
    epsilon = float(_attr(kwargs, "epsilon", 1e-6))

    x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]

    # A2 权威语义(对齐 canndev normal/single_row_kernel:逐个 Cast fp32 再 Add,全程 fp32):
    # 注释"1.将x1的第一个tensor和x2相加 2.将x1剩余的相加" => x = (x1[0]+x2) + x1[1]+...+x1[n-1]
    x_fp32 = x1_list[0].to(torch.float32) + x2.to(torch.float32)
    for t in x1_list[1:]:
        x_fp32 = x_fp32 + t.to(torch.float32)
    x_out_t = x_fp32.to(x1_list[0].dtype)

    var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + epsilon)
    normed = x_fp32 * rstd * gamma.to(torch.float32)
    y_out_t = normed.to(x1_list[0].dtype)

    def quant(smooth):
        if smooth is not None:
            q = normed * smooth.to(torch.float32)
        else:
            q = normed
        scale = q.abs().amax(dim=-1, keepdim=True) / 127.0
        yq = torch.round(q / scale)  # round-half-to-even,对齐 CAST_RINT
        yq = torch.clamp(yq, -128, 127).to(torch.int8)
        return yq, scale.squeeze(-1).to(torch.float32)

    y1_t, scale1_t = quant(smooth_scale1)
    if smooth_scale2 is not None:
        y2_t, scale2_t = quant(smooth_scale2)
    else:
        y2_t, scale2_t = None, None
    return [y1_t, y2_t, x_out_t, y_out_t, scale1_t, scale2_t]


class _MarndqCompose:
    """三方标杆：torch 拼接在远端 GPU 执行。参数名与 def.cpp 逐字一致
    （x1/x2/gamma/smooth_scale1/smooth_scale2 + attr epsilon）。

    ⚠️ 竞品最优形态（性能对标铁律）：torch.compile 融合（eager 逐算子 launch 全是开销，
    NPU 侧是单个融合内核，拿未融合 eager 比不是同一层次）；编译失败回落 eager。
    ⚠️ x/y 输出 cast 回 NPU 的输出 dtype（fp16/bf16）——竞品留在 fp32 会让 cross_check
    的 ratio 凭空爆表（gnsq 实测 mare 961→1.0 的教训）。
    smoothNum<2 时 y2/scale2 返回空占位（golden 侧 None 哨兵使该输出整列跳过比对，
    三方值不会被读取；空 tensor 仅为远端序列化安全）。
    """

    def __init__(self, *, epsilon=1e-6, **_):
        self.epsilon = float(epsilon)
        self._compiled = None

    def _impl(self, x1, x2, gamma, smooth_scale1=None, smooth_scale2=None):
        x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
        x_fp32 = x1_list[0].to(torch.float32) + x2.to(torch.float32)
        for t in x1_list[1:]:
            x_fp32 = x_fp32 + t.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.epsilon)
        normed = x_fp32 * rstd * gamma.to(torch.float32)

        def quant(smooth):
            q = normed if smooth is None else normed * smooth.to(torch.float32)
            scale = q.abs().amax(dim=-1, keepdim=True) / 127.0
            yq = torch.clamp(torch.round(q / scale), -128, 127).to(torch.int8)
            return yq, scale.squeeze(-1).to(torch.float32)

        y1, scale1 = quant(smooth_scale1)
        x_out = x_fp32.to(x1_list[0].dtype)
        y_out = normed.to(x1_list[0].dtype)
        if smooth_scale2 is not None:
            y2, scale2 = quant(smooth_scale2)
        else:
            y2 = torch.zeros((0,), dtype=torch.int8, device=x2.device)
            scale2 = torch.zeros((0,), dtype=torch.float32, device=x2.device)
        return [y1, y2, x_out, y_out, scale1, scale2]

    def __call__(self, x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **_):
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(x1, x2, gamma, smooth_scale1, smooth_scale2)
        except Exception:
            return self._impl(x1, x2, gamma, smooth_scale1, smooth_scale2)


class MultiAddRmsNormDynamicQuantSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray（x1 为 list[np.ndarray]）、返 list。"""

    def golden(x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs):
        x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
        in_dtype_name = x1_list[0].dtype.name
        outs = _compute(
            [_to_torch(a) for a in x1_list],
            _to_torch(x2),
            _to_torch(gamma),
            _to_torch(smooth_scale1) if smooth_scale1 is not None else None,
            _to_torch(smooth_scale2) if smooth_scale2 is not None else None,
            **kwargs,
        )
        y1_t, y2_t, x_t, y_t, scale1_t, scale2_t = outs
        y1 = y1_t.numpy()
        y2 = y2_t.numpy() if y2_t is not None else None
        x = _from_torch(x_t, in_dtype_name)
        y = _from_torch(y_t, in_dtype_name)
        scale1 = scale1_t.numpy()
        scale2 = scale2_t.numpy() if scale2_t is not None else None
        return [y1, y2, x, y, scale1, scale2]

    third_party = {"torch": _MarndqCompose}
    tolerance = _TOL


def multi_add_rms_norm_dynamic_quant_golden(
    x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs
):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(
        MultiAddRmsNormDynamicQuantSpec.golden(
            x1, x2, gamma, smooth_scale1, smooth_scale2, **kwargs
        )
    )


__spec__ = {"multi_add_rms_norm_dynamic_quant": "MultiAddRmsNormDynamicQuantSpec"}
__golden__ = {
    "kernel": {
        "multi_add_rms_norm_dynamic_quant": "multi_add_rms_norm_dynamic_quant_golden"
    }
}

# 【不存在】aclnn / e2e / tf / onnx 通路：A2 仅 GE 图 + 融合 pass（无 op_api/torch 绑定/
# 框架插件），A5 对齐不新增（01_requirement.md §3.3：通路 = GE 图 + 融合 pass）。
