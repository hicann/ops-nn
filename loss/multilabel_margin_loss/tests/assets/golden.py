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
TTK TestSpec for multilabel_margin_loss (kernel / GEIR / aclnn / e2e 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，竞品接口 torch.ops.aten.multilabel_margin_loss_forward
                   （aten forward 即权威参照，非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一竞品 aten 接口在远端 GPU 上执行
    tolerance    —— 浮点输出 y 走 cross_check；is_target(int32) 承载掩码语义，
                   差 1 就是错，保持 binary_equal（不套 ±1 容忍）

Canonical IO order (multilabel_margin_loss_def.cpp):
    inputs : x(fp16/bf16/fp32), target(int32)
    outputs: y(同 x dtype), is_target(int32)
    attrs  : reduction(OPTIONAL str="mean", in {none, sum, mean})

aclnn 通路参数名取 aclnn_multilabel_margin_loss.h（self/target），reduction 为 int64
（0=none/1=mean/2=sum），golden 内映射回字符串。e2e 通路 torch.nn.functional.multilabel_margin_loss
只返回 loss（is_target 是 NPU 内部输出，e2e 不暴露），故 TorchSpec 只产 y。

aten forward 要求 target 为 int64（LongTensor），golden/三方统一由 int32 转 int64 再调；
fp16/bf16 在 CPU 侧 aten 不支持，抬 fp32 计算再舍回（与 NPU 内核 fp32 累加一致，不抬 fp64）。
"""

import numpy as np
import torch

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
    "int32": {"standard": "binary_equal"},
}

_REDUCTION_STR2INT = {"none": 0, "mean": 1, "sum": 2}
_REDUCTION_INT2STR = {0: "none", 1: "mean", 2: "sum"}


def _attr(kwargs, name, default):
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
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_INT2STR[int(v)]


def _compute(x, target, **kwargs):
    """torch.Tensor 进 / 出，返回 [y, is_target]，顺序照 def.cpp 输出序。

    fp16/bf16 输入抬 fp32 计算（aten CPU 不支持 half；与 NPU 内核 fp32 累加一致）。
    返回的 y 保持计算精度，舍回 x dtype 由各路壳负责。
    """
    reduction = _reduction_str(_attr(kwargs, "reduction", "mean"))
    # golden 在 **float64** 上算: 它是仲裁真值, 必须比被测两腿都准。
    # kernel/aclnn 通路有 golden_mode=Promote 抬到 fp64, 这里是空操作;
    # **e2e 通路的 Promote 实测不生效**(framework_api/golden_generation.py 自带
    # "flat_dtypes 取不到就原样返回、Promote 空转" 的静默路径), golden 停在 fp32 后会与
    # 同精度的三方腿算出逐位相同的值 → |三方-golden|=0 → cross_check 比值分母塌陷判红。
    # MSELossV2 实测两例红即此因, 改 fp64 后 e2e 240/240 全通。
    # aten.multilabel_margin_loss_forward 已实测支持 fp64(CPU)。
    xf = x.to(torch.float64)
    if x.numel() == 0 and reduction != "none":
        # [0,C] 的 sum/mean：**不能用 aten 当真值** —— 它在空输入上读的是未初始化内存
        # （同一输入连续调用会得到 3.6e19 / 1.6e-12 等互不相同的值）。取归约的通用约定：
        # sum(空)=0、mean(空)=0/0=nan（与 torch.empty(0).mean() 一致），与算子实现所取标准同源。
        y = torch.tensor(
            0.0 if reduction == "sum" else float("nan"),
            dtype=xf.dtype,
            device=xf.device,
        )
        return [y, torch.zeros_like(target, dtype=torch.int64)]
    out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
        xf, target.to(torch.int64), _REDUCTION_STR2INT[reduction]
    )
    return [out, is_target]


def _to_torch(a):
    """numpy → torch，原样保留 dtype；bf16(ml_dtypes) 按位 view（无损机械转换）。"""
    arr = np.ascontiguousarray(a)
    if _bf16 is not None and arr.dtype == _bf16:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _cast_np(t, target_dt):
    """torch 计算结果 → numpy 目标 dtype（bf16 经 ml_dtypes）。"""
    arr = (
        t.detach().cpu().to(torch.float32).numpy()
        if t.dtype == torch.bfloat16
        else t.detach().cpu().numpy()
    )
    if target_dt == "bfloat16":
        return arr.astype(_bf16) if _bf16 is not None else arr
    return arr.astype(target_dt)


# 三方腿的 is_target 用**1 元素占位**回传, 不回传整块。
#
# 依据: is_target 恒为 int32, _TOL["int32"] = binary_equal, 而 BinaryComparison.compare_impl
# 只用 self.output 与 self.golden(见 ttk/core_modules/comparison/binary_equal.py) ——
# **三方腿的 is_target 值从未被任何判据使用**。而它的尺寸是 N x C, 比 y 大几个数量级:
#   (362038, 11) 单例: y(sum)=4B / y(none)=1.4MB / is_target=15.9MB
# 这条链路是 proxyjump 公网跳板(:80), 实测吞吐仅 85 KB/s。白传的 is_target 既拖慢跑批,
# 又把下行堆到上百 MB 触发跳板的连接限制 —— ssh -vvv 实测:
#   "Connection to 127.0.0.1 closed by remote host.
#    Transferred: sent 169724, received 148991832 bytes, in 1749.4 seconds"
# 断连后端点消失 -> 三方腿连续失败 -> 熔断器中止整批(实测 244/300、50/318 两次停摆)。
#
# 输出**个数必须保持 2**: comparison.py 有 third_party_count_mismatch 校验, 少返回会直接判错。
# 占位符 dtype 与真实输出一致, 只把元素数降到 1。
_IST_PLACEHOLDER_NOTE = True


class _MmlCompose:
    """三方标杆：aten forward 在远端 GPU 执行。参数名与 def.cpp 逐字一致（x/target/reduction）。

    GPU 上抬 fp32 计算再舍回 x dtype（half CUDA 支持度不一，且竞品留在 fp32 会让
    cross_check ratio 凭空爆表——输出必须 cast 回 NPU 的输出 dtype）。
    """

    def __init__(self, *, reduction="mean", **_):
        self.reduction = _reduction_str(reduction)

    def __call__(self, x, target, **_):
        out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
            x.to(torch.float32),
            target.to(torch.int64),
            _REDUCTION_STR2INT[self.reduction],
        )
        # is_target 回传 1 元素占位, 不回传整块 —— 见文件上方 _IST_PLACEHOLDER_NOTE:
        # 它的判据是 binary_equal(NPU 对 golden 两腿), 三方腿的值从不被使用, 而尺寸是 N x C。
        # golden 壳必须保留完整(binary_equal 要拿它跟 NPU 比), 只有三方腿这侧可以省。
        return [out.to(x.dtype), is_target.reshape(-1)[:1].to(torch.int32)]


class _MmlAclnnCompose:
    """aclnn 通路三方标杆（aclnnMultilabelMarginLoss）：输入名取 aclnn 头文件
    （self/target），reduction 为 int64(0=none/1=mean/2=sum)；is_target 跟随 self dtype
    （A5 aclnn 契约）。`self` 仅位置参数防撞名（acl nn 首参名即 self，
    服务端按名绑定时关键字 self 会撞实例参数，positional-only 后落入 **kw）。"""

    def __init__(self, /, reduction=1, **_):
        self._red = _reduction_str(reduction)

    def __call__(self, /, *args, **kw):
        """入参按 aclnn 头文件顺序: (self, target, reduction, out, isTarget)。

        **位置与关键字都要接**: TTK 对 aclnn 三方腿是按头文件形参**顺序**下发的
        (参照 activation/elu_grad_v2 的 aclnn golden —— 连标量和输出张量都在形参里)。
        只声明 `**kw` 时 args 为空、kw 里没有 self/input, 取到 None 后 `.to()` 直接崩,
        服务端返回 500 AttributeError, 熔断器连续 30 例失败后中止整批 ——
        MSELossV2 的 aclnn 通路实测踩过这个坑(240 例只收 29 条)。
        """
        x = args[0] if len(args) > 0 else kw.get("self", kw.get("input"))
        tgt = args[1] if len(args) > 1 else kw.get("target")
        red = args[2] if len(args) > 2 else kw.get("reduction", self._red)
        out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
            x.to(torch.float32),
            tgt.to(torch.int64),
            _REDUCTION_STR2INT[_reduction_str(red)],
        )
        # 这里**不能**用 1 元素占位(kernel 通路那侧可以): aclnn 契约下 is_target 的 dtype
        # 跟随 x(fp32/fp16/bf16), 命中 _TOL 的浮点行 -> 判据是 cross_check, **要读三方腿**;
        # kernel 通路的 is_target 恒 int32 -> binary_equal, 三方腿是死代码才可省。
        # 实测证据: 套了占位后 171 例里只有 C=1 的 11 例过(占位恰好等于完整张量),
        # C>1 的 160 例全报 COMPARE_FAILURE 且 precision_metrics 为空。
        return [out.to(x.dtype), is_target.to(x.dtype)]


class _MmlTorchCompose:
    """e2e 通路三方标杆（torch.nn.functional.multilabel_margin_loss）：输入名取 torch API
    签名（input/target）。e2e 只产 y（API 不暴露 is_target）。"""

    def __init__(self, *, reduction="mean", **_):
        self._red = _reduction_str(reduction)

    def __call__(self, input, target, **_):
        out, _is_target = torch.ops.aten.multilabel_margin_loss_forward(
            input.to(torch.float32),
            target.to(torch.int64),
            _REDUCTION_STR2INT[self._red],
        )
        return [out.to(input.dtype)]


def _inject_nonfinite(x, target, testcase_name=""):
    """错误场景档 nonfinite 的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(ttk/utilities/data.py `_digitize_inf_nan`), 写了也造不出
    非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入 x(浮点数据面), **不注入 target** —— target 是 int32 标签下标,
    往里写非有限数会变成越界下标, 那验的是"越界拒收"而不是"非有限值传播", 两码事。
    位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan。元素数 < 4 不注入。

    预期行为: 按 IEEE 语义自然传播(对齐 torch), 仍走正常精度判据, 不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return x, target
    a = np.asarray(x).copy()
    if a.size < 4:
        return x, target
    f = a.reshape(-1)
    f[0], f[1], f[2] = np.inf, -np.inf, np.nan
    return a, target


class MultilabelMarginLossKernelSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 [y, is_target]。"""

    def golden(x, target, **kwargs):
        outs = _compute(_to_torch(x), _to_torch(target), **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        y_dt = od[0] if len(od) > 0 else str(np.asarray(x).dtype)
        t_dt = od[1] if len(od) > 1 else "int32"
        return [_cast_np(outs[0], y_dt), _cast_np(outs[1], t_dt)]

    def customize_inputs(x, target, **kwargs):
        return _inject_nonfinite(x, target, kwargs.get("testcase_name", ""))

    third_party = {"torch": _MmlCompose}
    tolerance = _TOL


class MultilabelMarginLossAclnnSpec:
    """aclnn 通路 spec：golden 收已 H2D 的 torch.Tensor。

    参数名取 aclnn_multilabel_margin_loss.h（self/target），reduction 为 int64
    （0=none/1=mean/2=sum）。首参名 self 照写：TTK 从类取普通函数，不绑定实例。
    """

    def golden(self, target, reduction=1, out=None, isTarget=None, **kwargs):
        # 签名与 aclnnMultilabelMarginLossGetWorkspaceSize 一致(不含 workspaceSize/executor);
        # reduction 为 int64(0=none/1=mean/2=sum)。
        outs = _compute(self, target, reduction=reduction, **kwargs)
        # **y 不得降回 self.dtype**: golden 是仲裁真值, 砍回输入精度后与同精度的三方腿
        # 可能逐位相同 → |三方-golden|=0 → cross_check 比值分母塌陷判红; 大值域下还会
        # 自己先溢出。MSELossV2 实测两例红即此因(golden 与三方同为 2437.4717 / golden=inf)。
        # 比对器会按需提升类型, 返回高精度是安全的。
        # is_target 是 0/1 掩码, 其 dtype 属**契约**(A5 aclnn 跟随 self)而非精度决策, 保留 cast。
        return [outs[0], outs[1].to(self.dtype)]

    # TTK ≥193da3e 起 aclnn 通路支持三方;指向 aclnn 名参数版 compose(首参 self 防撞名)。
    third_party = {"torch": _MmlAclnnCompose}
    tolerance = _TOL


class MultilabelMarginLossTorchSpec:
    """e2e 通路 spec（torch.nn.functional.multilabel_margin_loss）：只产 y（API 不暴露 is_target）。

    ⚠️ 签名按 torch API 全参数序：框架按 param plan **位置**传全量参数
    （input, target, size_average, reduce, reduction），少声明会报参数数错。
    """

    def golden(input, target, *args, **kwargs):
        # 框架按 param plan 位置传全量参数(docstring 解析含 out,共 6 个)。
        # 稳健取 reduction:位置参数里的合法字符串,或 kwargs;缺省 mean。
        reduction = kwargs.get("reduction")
        if reduction is None:
            strs = [
                a for a in args if isinstance(a, str) and a in ("none", "mean", "sum")
            ]
            reduction = strs[0] if strs else "mean"
        outs = _compute(input, target, reduction=reduction, **kwargs)
        # **不得降回 input.dtype**: e2e 通路的 golden Promote 实测不生效
        # (framework_api/golden_generation.py 自带 "flat_dtypes 取不到就原样返回" 的静默路径),
        # 再砍回输入精度后 golden 与同精度三方腿可能逐位相同 → 比值分母塌陷判红,
        # 大值域下还会自己先溢出。MSELossV2 的 e2e 两例红即此因, 改回高精度后 240/240 全通。
        return [outs[0]]

    third_party = {
        "torch": _MmlTorchCompose
    }  # e2e 名(input/target),与 kernel/aclnn 版区分
    tolerance = _TOL


def multilabel_margin_loss_golden(x, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(MultilabelMarginLossKernelSpec.golden(x, target, **kwargs))


__spec__ = {
    "multilabel_margin_loss": "MultilabelMarginLossKernelSpec",
    "aclnnMultilabelMarginLoss": "MultilabelMarginLossAclnnSpec",
    "torch.nn.functional.multilabel_margin_loss": "MultilabelMarginLossTorchSpec",
}
__golden__ = {"kernel": {"multilabel_margin_loss": "multilabel_margin_loss_golden"}}

# 【不存在】tf / onnx 通路：tf_plugin 无本体（仅 shape 变体）、onnx 无（01_requirement.md §3.3）。
# 注：torch 图模式经 GE 图通路到达，不单独注册（同 geir）。
