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
TTK TestSpec for mse_loss_v2 (kernel / GEIR / aclnn / e2e 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，竞品接口 F.mse_loss（非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一竞品接口在远端 GPU 上执行（cross_check 比值的另一腿）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

Canonical IO order (mse_loss_v2_def.cpp):
    inputs : input, target
    outputs: output
    attrs  : reduction(OPTIONAL str="mean", in {none, sum, mean})

kernel 内部 fp16/bf16->fp32 计算再 RNE 舍回，golden 同样抬 fp32 计算、末位照 output_dtypes
舍回（中间精度与算子实现一致，不抬 fp64）。

aclnn 通路复用 loss/mse_loss 的 aclnnMseLoss（soc∈{910b,910_93,310p} 且 self.shape==target.shape
且 ND/NCL 时 dispatch 到 MSELossV2）：参数名取 aclnn_mse_loss.h（self/target），reduction 为
int64（0=none/1=mean/2=sum，aclnn_mse_loss.cpp REDUCTION_*_NUM），golden 内映射回字符串。
e2e 通路即 torch.nn.functional.mse_loss（torch_npu 经 PrivateUse1 落到 aclnnMseLoss）。
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_REDUCTION_NUM2STR = {0: "none", 1: "mean", 2: "sum"}


def _attr(kwargs, name, default):
    """attributes 可能平铺在 kwargs，也可能收在 kwargs['attributes'] dict；字符串做类型归一。"""
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "false", "yes", "no", "1", "0") and isinstance(default, bool):
            return s in ("true", "yes", "1")
        try:
            return type(default)(v)
        except Exception:
            return default
    return v


def _reduction_str(v):
    """reduction 归一到字符串：kernel/e2e 通路本来就是 str，aclnn 通路是 int64。"""
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_NUM2STR[int(v)]


def _f32_floor(t):
    """fp16/bf16 抬 fp32（CPU half 支持残缺，且与 NPU 内核 fp32 计算一致）；
    fp32/fp64 照单全收——cross_check 场景框架按 golden_mode=Promote 自动把输入抬一档
    （fp32→fp64），golden 不自行 cast、不替框架做精度决策（砍回 fp32 会废掉 Promote）。
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _compute(input, target, **kwargs):
    """torch.Tensor 进 / 出，返回 list[Tensor]（舍回输出 dtype 由各路壳负责）。

    归约在 **float64** 上做: golden 是仲裁真值, 必须比被测两腿都准。
    kernel/aclnn 通路有 golden_mode=Promote 把输入抬到 fp64, 这里是空操作;
    **e2e 通路的 Promote 实测没生效**(framework_api/golden_generation.py 自带一条
    "flat_dtypes 取不到就原样返回、Promote 空转" 的静默路径), golden 停在 fp32 后:
      · ±1e18 档先求和再除会溢出 → golden 自己变 inf(实测 e2e_0055);
      · 与同为 fp32 的三方腿算出**逐位相同**的值 → |三方-golden|=0, 比值分母塌陷
        → rmse 比值虚高判红(实测 e2e_0235, golden 与三方同为 2437.4717)。
    两种都不是 NPU 的问题, 是参照腿失去了仲裁能力。
    """
    reduction = _reduction_str(_attr(kwargs, "reduction", "mean"))
    x64 = _f32_floor(input).to(torch.float64)
    t64 = _f32_floor(target).to(torch.float64)
    y = F.mse_loss(x64, t64, reduction=reduction)
    return [y]


def _to_torch(arr):
    """numpy → torch，原样保留 dtype（fp64/fp32 直转；fp16 直转；
    bf16 经 ml_dtypes 按位 view，无损机械转换，非精度决策）。"""
    a = np.ascontiguousarray(arr)
    if _bf16 is not None and a.dtype == _bf16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _third_party_mse(input, target, reduction):
    """三方腿的 mse：按 §五 四格对照表「浮点 + 三方」那一行实现。

    规则原文：**GPU（三方腿）按 NPU 加宽行为同步 cast，出口对应窄回**。
    此前这里是直接 `F.mse_loss(input, target)` 按原 dtype 算，两处不符：

    ① **没按 NPU 加宽**。本算子 fp16/bf16 输入内部提升 fp32（03_spec.yaml
       `intermediate_dtype: float32`），三方腿却在 fp16 上累加 —— 实测 q0170
       (fp16, ±180, 1048576 元素) 累加和 2.29e10 远超 fp16 上限 65504，三方腿返回 inf，
       而 NPU 与 fp64 golden 都是 21792.0。
    ② **mean 的运算序列与 NPU 不等价**。NPU 在求和完成前就除以 N；三方腿先求和再除，
       中间和会涨到 N 倍 —— 实测 q0178 (fp32, ±1e18, 14512 元素) 中间和 1.43e40 超
       fp32 上限 3.4e38，任何"先求和再除"的 fp32 实现都必然 inf，而真值 9.85e35 在
       fp32 完全可表示，NPU 也算出来了。

    两处都会让三方腿吐 inf，而 cross_check 是三腿比值判据，分母 |三方−golden| 无定义
    → mare/mere/rmse 全 None、整条用例判不了。**这是参照腿的缺陷，不是算子的**。

    ⚠ 独立性未破（§六之二 V1 / lamb 七算子那个坑）：加宽只改精度落点、不改 torch 自己的
    归约顺序，实测 fp32 档误差仍在 1e-8~1e-7 且多数用例与改前不同值，fp16 档误差 ~1e-4
    由输出 cast 主导 —— 不是"两腿同构、三比值恒 1.0"的退化。
    """
    out_dt = input.dtype
    # 按 NPU 的加宽行为: fp16/bf16 → fp32; fp32/fp64 保持原样(不得反向窄化)
    work = torch.float32 if out_dt in (torch.float16, torch.bfloat16) else out_dt
    a, b = input.to(work), target.to(work)
    d = (a - b) ** 2
    red = _reduction_str(reduction)
    if red == "mean":
        # 先除 N 再累加, 与 NPU 的归约序列对齐(NPU 在跨核合并前已除 N)
        d = d / torch.tensor(float(d.numel()), dtype=work, device=d.device)
        out = d.sum()
    elif red == "sum":
        out = d.sum()
    else:
        out = d
    return out.to(out_dt)


class _MseLossCompose:
    """三方标杆：竞品接口 F.mse_loss 直出，由 TTK 派发到远端 GPU 执行。

    参数绑定契约：属性喂 __init__、输入喂 __call__，参数名与 def.cpp 逐字一致
    （input / target / reduction）。输出 dtype 天然随输入（= NPU 输出 dtype），无需额外 cast。
    """

    def __init__(self, *, reduction="mean", **_):
        self.reduction = _reduction_str(reduction)

    def __call__(self, input, target, **_):
        return [_third_party_mse(input, target, self.reduction)]


class _MseLossAclnnCompose:
    """aclnn 通路三方标杆（aclnnMseLoss）：输入名取 aclnn 头文件（self/target）、
    reduction 为 int64(0=none/1=mean/2=sum)。

    ⚠️ `self` 仅位置参数：aclnn 首参名就叫 self，服务端按名绑定时 self=<tensor>
    会以关键字进 **_；若方法的 self 是常规位置关键字参数，调用即撞名
    "got multiple values for argument 'self'"。positional-only 声明后，
    关键字 self 只能落进 **kw，不撞实例参数。__init__/__call__ 都同理要加。
    """

    def __init__(self, /, reduction=1, **_):
        self._red = _reduction_str(reduction)

    def __call__(self, /, *args, **kw):
        """入参按 aclnn 头文件顺序: (self, target, reduction, out)。

        **位置与关键字都要接**: TTK 对 aclnn 三方腿是按头文件形参**顺序**下发的
        (参照 activation/elu_grad_v2 的 aclnn golden —— 连标量和输出张量都在形参里),
        原来只写 `**kw` 时 args 为空、kw 里没有 self/input, 取到 None 后 `.dtype` 直接崩,
        服务端返回 500 AttributeError, 熔断器连续 30 例失败后中止整批(实测)。
        """
        x = args[0] if len(args) > 0 else kw.get("self", kw.get("input"))
        t = args[1] if len(args) > 1 else kw.get("target")
        red = args[2] if len(args) > 2 else kw.get("reduction", self._red)
        return [_third_party_mse(x, t, red)]


def _inject_nonfinite(input, target, testcase_name=""):
    """DFX-nonfinite 档的定点注入：只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan：TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**（ttk/utilities/data.py `_digitize_inf_nan`），
    写了也造不出非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    注入位置固定（首元素 +inf、次元素 -inf、第三个 nan，各自在 input/target 上错开），
    不随机：随机会让复现依赖 seed，且小 shape 时可能一个都注不进去。
    元素数 < 4 的用例不注入（位置放不下，注入会把整个张量变成非有限，失去"传播"这一档的意义）。

    预期行为见 03_spec.yaml `dfx_expect.nonfinite`：NaN/Inf 无特判，按 IEEE 语义自然传播
    （对齐 A2 与 torch），因此仍走正常精度判据，不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return input, target
    a, b = np.asarray(input).copy(), np.asarray(target).copy()
    if a.size < 4:
        return input, target
    fa, fb = a.reshape(-1), b.reshape(-1)
    # bf16 是 ml_dtypes 扩展类型, np.inf 可直接赋值（按 IEEE 编码写入），无需绕 view
    fa[0], fa[2] = np.inf, np.nan
    fb[1] = -np.inf
    return a, b


class MseLossV2KernelSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray]。"""

    def golden(input, target, **kwargs):
        outs = _compute(_to_torch(input), _to_torch(target), **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        target_dt = od[0] if od else str(np.asarray(input).dtype)
        out = outs[0].detach().cpu().numpy()
        if target_dt == "bfloat16":
            out = out.astype(np.float32).astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target_dt)
        return [out]

    def customize_inputs(input, target, **kwargs):
        return _inject_nonfinite(input, target, kwargs.get("testcase_name", ""))

    third_party = {"torch": _MseLossCompose}
    tolerance = _TOL


class MseLossV2AclnnSpec:
    """aclnn 通路 spec：golden 收已 H2D 的 torch.Tensor、返 Tensor。

    参数名取 op_api/aclnn_mse_loss.h（self/target），reduction 为 int64（0=none/1=mean/2=sum）。
    首参名 self 照写：TTK 用 getattr(cls,"golden") 从类取出，普通函数不绑定实例。
    """

    def golden(self, target, reduction=1, out=None, **kwargs):
        # 签名与 aclnnMseLossGetWorkspaceSize 一致(不含 workspaceSize/executor);
        # reduction 为 int64(0=none/1=mean/2=sum),golden 内映射回字符串。
        y = _compute(self, target, reduction=reduction, **kwargs)[0]
        # 与 TorchSpec 同理: **不得降回输入 dtype**。golden 是仲裁真值, 砍回 fp32 后
        # 与同为 fp32 的三方腿可能逐位相同 → |三方-golden|=0 → 比值分母塌陷判红
        # (实测 aclnn/e2e 同一例 0235_sum: golden 与三方同为 2437.4717)。
        return [y]

    # TTK ≥193da3e 起 aclnn 通路支持三方;指向 aclnn 名参数版 compose(首参 self 防撞名)。
    third_party = {"torch": _MseLossAclnnCompose}
    tolerance = _TOL


class MseLossV2TorchSpec:
    """e2e 通路 spec（torch.nn.functional.mse_loss）：golden 收已 H2D 的 torch.Tensor。

    ⚠️ 签名按 torch API 全参数序：框架按 param plan **位置**传全量参数
    （input, target, size_average, reduce, reduction），少声明会报"6 were given"。
    """

    def golden(input, target, *args, **kwargs):
        # 框架按 param plan 位置传全量参数(input, target, out, size_average, reduce,
        # reduction —— docstring 解析含 out,共 6 个)。稳健取 reduction:位置参数里的
        # 合法字符串,或 kwargs;缺省 mean。
        reduction = kwargs.get("reduction")
        if reduction is None:
            strs = [
                a for a in args if isinstance(a, str) and a in ("none", "mean", "sum")
            ]
            reduction = strs[0] if strs else "mean"
        y = _compute(input, target, reduction=reduction, **kwargs)[0]
        # **不得降回 input.dtype**: 那会把 fp64 仲裁值砍成 fp32, 与三方腿同精度,
        # 判据随即失去判别力(golden 必须比被测两腿都准 —— TTK Promote 的本意)。
        # 比对器会按需提升类型, 这里返回高精度是安全的。
        return [y]

    third_party = {"torch": _MseLossCompose}  # 【预留】同 AclnnSpec，当前不被取用
    tolerance = _TOL


def mse_loss_v2_golden(input, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return MseLossV2KernelSpec.golden(input, target, **kwargs)


__spec__ = {
    "mse_loss_v2": "MseLossV2KernelSpec",
    "aclnnMseLoss": "MseLossV2AclnnSpec",
    "torch.nn.functional.mse_loss": "MseLossV2TorchSpec",
}
__golden__ = {"kernel": {"mse_loss_v2": "mse_loss_v2_golden"}}

# 【不存在】tf / onnx 通路：无 framework 插件（01_requirement.md §3.3）。
# 注：torch 图模式经 GE 图通路到达，不单独注册（同 geir）。
