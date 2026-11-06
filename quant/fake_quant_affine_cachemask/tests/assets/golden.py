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

"""TTK Golden for FakeQuantAffineCachemask（独立参考实现）。

参考口径：主参考为显式公式（fp32 求商 + RINT 取整 + 饱和，与 docs 公式及 kernel
的 fp32 直接除法同口径），有限值与特殊值统一路由，保证同一元素的参考值不随张量
内其他元素变化；ATen 官方入口降为 third_party 交叉对照——其 CPU 实现在半整数商
处与直接除法差 1 个量化级（倒数乘法差异，tie 域已知分歧）。

特殊值契约（docs「特殊值语义」，与 PyTorch 对齐）：
  - NaN/−Inf：out = (quant_min − zero_point)·scale，mask = False
  - +Inf：out = (quant_max − zero_point)·scale，mask = False

ATen 实测语义（本机 torch CPU，2026-10-29）：
  - 取整次序按 zero_point dtype 分支：int32 zp 为两步式 round(x/s)+zp；
    float/half zp 为一步式 lrint(x/s+zp)
  - NaN/±Inf：q 钳到 quant 边界、mask=False（与上述契约一致）
  - scale 仅接受 Float/BFloat16（fp16 需先升 fp32）；zp 接受 Int32/Float/Half；
    0 维 x 拒收（需展成 (1,) 再调）；bf16 输入在 CPU aten 存在实现缺陷（先升 fp32）

注册键：
  kernel/GEIR：<opInterface 小写名>（Host regular 投影 CSV 的 op_name 形态）与
              驼峰名并列注册
  aclnn：aclnnFakeQuantPerTensorAffineCachemask / aclnnFakeQuantPerChannelAffineCachemask
"""

import torch

__golden__ = {
    "kernel": {
        "fake_quant_affine_cachemask": "fake_quant_affine_cachemask_golden",
        "FakeQuantAffineCachemask": "fake_quant_affine_cachemask_golden",
    },
    "aclnn": {
        "aclnnFakeQuantPerTensorAffineCachemask": "aclnn_fake_quant_per_tensor_golden",
        "aclnnFakeQuantPerChannelAffineCachemask": "aclnn_fake_quant_per_channel_golden",
    },
}


def _to_numpy(t):
    """Torch → NumPy：bfloat16 经 uint16 位视图无损转 ml_dtypes.bfloat16 载体。"""
    if t.dtype == torch.bfloat16:
        import ml_dtypes

        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _bf16_to_torch(a):
    """ml_dtypes.bfloat16 ndarray → torch.bfloat16（经 fp32 无损往返）。"""
    import numpy as np

    return torch.from_numpy(a.astype(np.float32).copy()).to(torch.bfloat16)


def _expand_pt(x_t, s_t, z_t, axis):
    """PT（per-tensor）形态分发：scale/zero_point 为单元素时按 Host SelectMode
    的 MODE_PT 语义（整张量共用同一组量化参数）展开到 axis 维长度后走
    per-channel ATen——保留 zero_point dtype 的取整语义（ATen per-channel 原生
    支持浮点 zp，per-tensor ATen 仅接受 int zp）。"""
    if (
        s_t is None
        or z_t is None
        or x_t.numel() == 0
        or x_t.dim() == 0
        or s_t.numel() != 1
        or z_t.numel() != 1
    ):
        return s_t, z_t
    ax = int(axis) % x_t.dim()
    n = int(x_t.shape[ax])
    return s_t.reshape(1).repeat(n), z_t.reshape(1).repeat(n)


def fake_quant_affine_cachemask_golden(
    x, scale, zero_point, *, axis, quant_min, quant_max, **kwargs
):
    """Kernel/GEIR golden（参数名与顺序跟随 fake_quant_affine_cachemask_def.cpp，不含输出）。

    输入为 numpy.ndarray。主参考统一为显式公式（fp32 求商 + RINT 取整 + 饱和，
    与 docs 公式和 kernel 的 fp32 直接除法同口径）；ATen 降为 third_party
    交叉对照——其 CPU 实现在半整数商处与直接除法差 1 个量化级（倒数乘法差异），
    作为主参考会使同一元素的期望值随张量内是否含 NaN 而翻转。
    """
    import numpy as np

    def _t(a):
        if isinstance(a, np.ndarray):
            if a.dtype.name == "bfloat16":  # ml_dtypes.bfloat16
                return _bf16_to_torch(a)
            # ascontiguousarray 会把 0 维升为 (1,)，破坏标量形状（rank-0 支持域）
            if a.ndim == 0:
                return torch.from_numpy(a.copy())
            return torch.from_numpy(np.ascontiguousarray(a).copy())
        if isinstance(a, (float, np.floating)):
            return torch.tensor(float(a))
        if a is None or isinstance(a, torch.Tensor):
            return a
        return torch.tensor(a)

    x_t, s_t, z_t = _t(x), _t(scale), _t(zero_point)
    # 空张量短路（与 kernel/aclnn 的空输入契约一致）：同形状空输出
    if x_t.numel() == 0:
        return [
            _to_numpy(x_t.clone()),
            _to_numpy(torch.ones(x_t.shape, dtype=torch.bool)),
        ]
    # 有限值/特殊值/超大商统一走显式公式：逐元素参考不随张量内其他元素变化
    # （单元素 scale/zero_point 经 bc 广播天然覆盖 PT 形态，无需展开）
    y, m = _explicit_formula(x_t, s_t, z_t, axis, quant_min, quant_max)
    if not torch.is_tensor(y):
        return list(y)
    return [_to_numpy(y), _to_numpy(m)]


def _explicit_formula(x, scale, zero_point, axis, quant_min, quant_max):
    """特殊值/超大商域显式公式：rint(Cast int32 饱和语义) + 契约化特殊值映射
    （NaN/−Inf→INT32_MIN→quant_min、+Inf→INT32_MAX→quant_max，mask=False，
    对齐 docs「特殊值语义」与 kernel 的 ForceNonFiniteToSentinel 哨兵路径）。"""
    orig_scalar = x.dim() == 0
    if orig_scalar:
        x = x.reshape(1)
    qmin, qmax = int(quant_min), int(quant_max)
    x_f = x.float()
    s_f = _to_fp32(scale)
    z_f = zero_point.to(torch.float32)
    moved = torch.moveaxis(x_f, axis, 0)
    bc = (-1,) + (1,) * (moved.dim() - 1) if moved.dim() > 0 else ()
    s_b = s_f.reshape(bc) if bc else s_f
    z_b = z_f.reshape(bc) if bc else z_f
    if torch.is_tensor(zero_point) and zero_point.dtype == torch.int32:
        t = moved / s_b  # 两步式（int32 zp）
        t = torch.nan_to_num(
            t, nan=-2147483648.0, posinf=2147483647.0, neginf=-2147483648.0
        )
        q = torch.round(t) + z_b
    else:
        t = moved / s_b + z_b  # 一步式（浮点 zp）
        t = torch.nan_to_num(
            t, nan=-2147483648.0, posinf=2147483647.0, neginf=-2147483648.0
        )
        q = torch.round(t)
    q = torch.clamp(q, -2147483648.0, 2147483647.0)
    q_c = torch.minimum(
        torch.tensor(float(qmax)), torch.maximum(torch.tensor(float(qmin)), q)
    )
    y = ((q_c - z_b) * s_b).to(x.dtype)
    m = torch.logical_and(q >= qmin, q <= qmax)
    if moved.dim() > 0:
        y = torch.moveaxis(y, 0, axis)
        m = torch.moveaxis(m, 0, axis)
    if orig_scalar:
        y = y.reshape(())
        m = m.reshape(())
    return y, m


def aclnn_fake_quant_per_channel_golden(
    self,
    scale,
    zeroPoint,
    axis=0,
    quantMin=-128,
    quantMax=127,
    out=None,
    mask=None,
    **kwargs,
):
    """ACLNN per-channel golden（输入 torch.Tensor，返回 list[torch.Tensor]）。

    主参考统一为显式公式（与 kernel golden 同口径，参考值不随张量内特殊值
    存在与否变化）；ATen 交叉对照走 third_party 入口。
    """
    if self.numel() == 0:
        return [self.clone(), torch.ones(self.shape, dtype=torch.bool)]
    y, m = _explicit_formula(
        self, scale, zeroPoint, int(axis), int(quantMin), int(quantMax)
    )
    return [y, m]


def aclnn_fake_quant_per_tensor_golden(
    self,
    scale,
    zeroPoint,
    fakeQuantEnabled=1.0,
    quantMin=-128,
    quantMax=127,
    out=None,
    mask=None,
    **kwargs,
):
    """ACLNN per-tensor golden（fakeQuantEnabled<1 时按文档语义输出 identity）。

    主参考统一为显式公式（与 kernel golden 同口径）。
    """
    # 兼容旧 CSV/头文件拼写（fakeQuantEnbled 缺 'a'）
    if "fakeQuantEnbled" in kwargs:
        fakeQuantEnabled = kwargs["fakeQuantEnbled"]
    if float(fakeQuantEnabled) < 1.0:
        return [self.clone(), torch.ones(self.shape, dtype=torch.bool)]
    if self.numel() == 0:
        return [self.clone(), torch.ones(self.shape, dtype=torch.bool)]
    y, m = _explicit_formula(self, scale, zeroPoint, 0, int(quantMin), int(quantMax))
    return [y, m]


# 显式 Spec 注册：Loader 按接口绑定 tolerance（bool 输出 binary_equal，浮点
# 输出 mix_tolerance），third_party 预留 ATen 交叉验证入口
__spec__ = {
    "fake_quant_affine_cachemask": "FakeQuantAffineCachemaskKernelSpec",
    "FakeQuantAffineCachemask": "FakeQuantAffineCachemaskKernelSpec",
    "aclnnFakeQuantPerTensorAffineCachemask": "AclnnFakeQuantPerTensorSpec",
    "aclnnFakeQuantPerChannelAffineCachemask": "AclnnFakeQuantPerChannelSpec",
}

_TOLERANCE = {
    "float16": {"standard": "mix_tolerance"},
    "float32": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "bool": {"standard": "binary_equal"},
}


class _AtenThirdParty:
    """third_party 交叉验证入口：按接口分发 per-channel/per-tensor ATen 调用。

    TTK 执行器按类实例化（cls(**kwargs) 后 cls(*args)）：__init__ 吸收执行器
    注入的命名参数，__call__ 收位置参数。per-tensor 接口额外读 fakeQuantEnabled。
    """

    def __init__(self, **kwargs):
        self._extra = kwargs

    def __call__(self, *args, **kwargs):
        import numpy as _np

        def _t(a):
            if isinstance(a, _np.ndarray):
                if a.dtype.name == "bfloat16":  # ml_dtypes.bfloat16
                    return _bf16_to_torch(a)
                return torch.from_numpy(_np.ascontiguousarray(a).copy())
            if isinstance(a, (float, _np.floating)):
                return torch.tensor(float(a))
            return a

        merged = dict(self._extra)
        merged.update(kwargs)
        if len(args) >= 3:
            x, scale, zp = _t(args[0]), _t(args[1]), _t(args[2])
            if x.numel() == 0:
                return [x.clone(), torch.ones(x.shape, dtype=torch.bool)]
            qmin = int(merged.get("quant_min", merged.get("quantMin", -128)))
            qmax = int(merged.get("quant_max", merged.get("quantMax", 127)))
            fqe = merged.get("fakeQuantEnabled", merged.get("fakeQuantEnbled", 1.0))
            if fqe is not None and float(fqe) < 1.0:
                return [x.clone(), torch.ones(x.shape, dtype=torch.bool)]
            axis = int(merged.get("axis", merged.get("Axis", 0)))
            # PT（per-tensor）形态：单元素 scale/zp 展开到 axis 维后走 per-channel
            # ATen（保留 zp dtype 取整语义），与 kernel golden 的分发同口径
            scale, zp = _expand_pt(x, scale, zp, axis)
            y, m = _aten_per_channel(x, scale, zp, axis, qmin, qmax)
            return [y, m]
        raise TypeError("AtenThirdParty expects (x, scale, zero_point) positional args")


class FakeQuantAffineCachemaskKernelSpec:
    golden = staticmethod(fake_quant_affine_cachemask_golden)
    third_party = {"torch": _AtenThirdParty}
    tolerance = _TOLERANCE


class AclnnFakeQuantPerChannelSpec:
    golden = staticmethod(aclnn_fake_quant_per_channel_golden)
    third_party = {"torch": _AtenThirdParty}
    tolerance = _TOLERANCE


class AclnnFakeQuantPerTensorSpec:
    golden = staticmethod(aclnn_fake_quant_per_tensor_golden)
    third_party = {"torch": _AtenThirdParty}
    tolerance = _TOLERANCE


_ATEN_PC = torch.ops.aten.fake_quantize_per_channel_affine_cachemask
_ATEN_PT = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask

# 16 位输入统一升 fp32 调 ATen（等价 kernel 的 Cast fp32 计算路径；ATen 对
# fp16 scale 拒收、bf16 输入存在 CPU 缺陷）
_FP16 = (torch.float16,)
_BF16 = (torch.bfloat16,)


def _to_fp32(t):
    if t is None:
        return t
    if t.dtype in (_FP16 + _BF16 + (torch.float64,)):
        return t.float()
    return t if t.dtype == torch.float32 else t.float()


def _aten_per_channel(x, scale, zero_point, axis, quant_min, quant_max):
    """ATen per-channel 独立参考：仅做 dtype/shape/axis 适配，语义完全交 ATen。"""
    orig_scalar = x.dim() == 0
    x2 = x.reshape(1) if orig_scalar else x
    x2 = _to_fp32(x2)
    scale2 = _to_fp32(scale)
    zp2 = zero_point
    if zp2 is not None:
        if zp2.dtype in (torch.int32, torch.float32, torch.float16):
            pass
        elif zp2.dtype in (torch.int64, torch.int16, torch.int8, torch.uint8):
            zp2 = zp2.to(torch.int32)  # 整型域保留（ATen 两步式分支）
        else:
            zp2 = zp2.to(torch.float32)  # bf16 zp 等升 fp32（ATen 拒收 bf16 zp）
    # ATen 仅接受 [0, ndim) 的正 axis：负 axis 归一化
    axis2 = int(axis) % x2.dim() if x2.dim() > 0 else 0
    # 空张量短路：kernel 对空输入不执行（aclnn 层 workspaceSize=0 直接返回），
    # golden 返回同形状空输出保持结构一致
    if x2.numel() == 0:
        return x2.to(x.dtype), torch.zeros(x2.shape, dtype=torch.bool)
    # ATen 前置约束 zp∈[quant_min, quant_max]：本算子允许越界 zp（q 钳位处理），
    # 越界形态退回显式公式（与 NaN/±Inf 同属 ATen 支持域外）
    if zp2 is not None and zp2.numel() > 0:
        zmin = zp2.min().item()
        zmax = zp2.max().item()
        if zmin < quant_min or zmax > quant_max:
            return _explicit_formula(x, scale, zero_point, axis, quant_min, quant_max)
    y, m = _ATEN_PC(x2, scale2, zp2, axis2, quant_min, quant_max)
    y = y.to(x.dtype)
    if orig_scalar:
        y = y.reshape(())
        m = m.reshape(())
    return y, m


def _aten_per_tensor(x, scale, zero_point, quant_min, quant_max):
    """ATen per-tensor 独立参考（scale 标量、zp int）。

    ATen 支持域外退显式公式：zp 越出 [quant_min, quant_max]（ATen 前置拒绝）、
    NaN/±Inf、超大商（int64 wrap 域）。
    """
    x2 = _to_fp32(x.reshape(1) if x.dim() == 0 else x)
    # 空张量短路（与 _aten_per_channel 同口径）：kernel/aclnn 对空输入不执行
    if x2.numel() == 0:
        return x2.to(x.dtype), torch.zeros(x2.shape, dtype=torch.bool)
    scale_v = float(scale.reshape(-1)[0]) if torch.is_tensor(scale) else float(scale)
    zp_t = zero_point if torch.is_tensor(zero_point) else torch.tensor(zero_point)
    zp_v = int(zp_t.reshape(-1)[0])
    # zp 越界：ATen 前置拒绝（zp 必须落在 quant 边界内）——退显式公式
    if not (quant_min <= zp_v <= quant_max):
        y, m = _explicit_formula(x, scale, zero_point, 0, quant_min, quant_max)
        return y, m
    # NaN/±Inf / 超大商域
    if torch.isnan(x2).any() or torch.isinf(x2).any():
        y, m = _explicit_formula(x, scale, zero_point, 0, quant_min, quant_max)
        return y, m
    if (
        abs(scale_v) < 1e-30
        or (x2.abs().max().item() / max(abs(scale_v), 1e-38)) > 4.6e18
    ):
        y, m = _explicit_formula(x, scale, zero_point, 0, quant_min, quant_max)
        return y, m
    y, m = _ATEN_PT(x2, scale_v, zp_v, quant_min, quant_max)
    y = y.to(x.dtype)
    if x.dim() == 0:
        y = y.reshape(())
        m = m.reshape(())
    return y, m
