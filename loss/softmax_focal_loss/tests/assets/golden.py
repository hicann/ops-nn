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
"""SoftmaxFocalLoss 多通路 golden(TestSpec 范式)。

通路支持表(照抄 01_requirement.md §3.3):
  | 通路   | 支持 | 依据                                                        |
  |--------|------|-------------------------------------------------------------|
  | kernel | ✅   | op_kernel/arch35/ 有实现                                     |
  | geir   | ✅   | op_graph/ 有 REG_OP(SoftmaxFocalLoss) + IMPL_OP_INFERSHAPE   |
  | aclnn  | ❌   | canndev 老树 built-in 与新树 ops/ 均无 op_api 实现            |
  | e2e    | ❌   | torch_npu 二进制无 aclnnSoftmaxFocalLoss 符号                 |
"""

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键(算子蛇形名), geir 不另写
    "softmax_focal_loss": "SoftmaxFocalLossKernelSpec",
}

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
# 输出 dtype 取自 op_host/softmax_focal_loss_def.cpp: {fp16, fp32}
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}

# 判据: 浮点输出配 cross_check 才会去取三方数据; L1 见 verification.md §5.2


def _attr(kwargs, name, default):
    """CSV 里的 attributes 可能是字符串, 统一转成 default 的类型。"""
    v = kwargs.get(name, default)
    if isinstance(v, str):
        try:
            return type(default)(v)
        except ValueError:
            return default
    return v


def _compute(pred, target, weight=None, **kwargs):
    """全程 torch.Tensor 进出, 返回 list[Tensor], 顺序照 def.cpp 的输出序。

    计算语义对齐 A2 tbe.dsl(softmax_focal_loss_compute):
        ce = -log(pred) * target * weight        CE = sum(ce, -1, keepdim)
        fw = alpha * exp(gamma * log(1 - pred)) * target
                                                 FW = sum(fw, -1, keepdim)
        y  = broadcast(CE * FW)                  # 整行同值
    reduction 在计算侧不生效(与 A2 同): 输出恒与 pred 同形。

    weight 缺省时按全 1 权重给出数学定义, 与 A2 实现(逐处判空后跳过加权)及 A5 内核
    的补齐行为一致。

    精度决策契约: cross_check 下框架按 Promote 把输入抬一档喂进来, 这里只向上兜底
    (half→fp32 防 CPU half 残缺), 绝不向下砍。
    """
    gamma = _attr(kwargs, "gamma", 2.0)
    alpha = _attr(kwargs, "alpha", 0.25)

    dt = pred.dtype
    if dt in (torch.float16, torch.bfloat16):
        dt = torch.float32
    p = pred.to(dt)
    t = target.to(dt)
    w = weight.to(dt) if weight is not None else torch.ones_like(p)

    neg_one = torch.tensor(-1.0, dtype=dt)
    one = torch.tensor(1.0, dtype=dt)
    g = torch.tensor(float(gamma), dtype=dt)
    a = torch.tensor(float(alpha), dtype=dt)

    # ce = -log(p) * t * w  →  CE
    ce = torch.mul(torch.log(p), neg_one)
    ce = torch.mul(ce, t)
    ce = torch.mul(ce, w)
    ce_sum = torch.sum(ce, dim=-1, keepdim=True)

    # fw = alpha * exp(gamma * log(1 - p)) * t  →  FW
    p_1sub = torch.add(torch.mul(p, neg_one), one)
    fw = torch.exp(torch.mul(torch.log(p_1sub), g))
    fw = torch.mul(fw, t)
    fw = torch.mul(fw, a)
    fw_sum = torch.sum(fw, dim=-1, keepdim=True)

    # y = CE * FW, 广播回整行
    res = torch.mul(ce_sum, fw_sum).expand_as(p).contiguous()
    return [res]


def _tp_widen(a):
    """三方腿按 NPU 的加宽行为同步 cast：**仅窄类型**抬 fp32（本算子内核 U=fp32），
    fp32 原样返回: 对计算类型与输入类型相同的档加 cast 是多余的,
    还会让竞品凭空变准、比值变大 → 假红。
    """
    return a.float() if a.dtype in (torch.float16, torch.bfloat16) else a


class _Compose:
    """竞品标杆(A100 上执行): 用 torch 高层表达式拼等价语义。

    与 _compute 的实现路径相互独立(这里走表达式/算子重载, _compute 走逐步 torch.mul),
    但算法保持一致: (1-p)^gamma 同样用 exp(gamma*log(1-p)) 而非 torch.pow ——
    pow 比 exp∘log 更准, 换写法会让竞品凭空更准约 2 倍, cross_check 假红。
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(self, pred, target, weight=None, **kwargs):
        out_dtype = pred.dtype
        narrow = pred.dtype in (torch.float16, torch.bfloat16)
        p = _tp_widen(pred)
        t = _tp_widen(target)
        w = _tp_widen(weight) if weight is not None else torch.ones_like(p)

        ce_sum = (-torch.log(p) * t * w).sum(dim=-1, keepdim=True)
        fw_sum = (self.alpha * torch.exp(self.gamma * torch.log(1.0 - p)) * t).sum(
            dim=-1, keepdim=True
        )
        y = (ce_sum * fw_sum).expand_as(p).contiguous()
        # 浮点输出必须 cast 回 NPU 输出 dtype, 否则竞品天然更准, ratio 失真
        return [y.to(out_dtype) if narrow else y]


def _inject_nonfinite(pred, testcase_name=""):
    """错误场景档 nonfinite 的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 `input_data_ranges` 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(`ttk/utilities/data.py` 的 `_digitize_inf_nan`), 写了也
    造不出非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入 pred(浮点数据面), **不动 target**(int32 one-hot 标签, 往里写非有限会变成
    非法标签, 测的是"非法标签"而不是"非有限值传播", 两码事)。
    资料依据(01_requirement.md §6): 「未校验 pred 值域…非法输入按 IEEE 语义传播, 不报错」
    与「p→1 时 log(1-p)→-inf」—— 本档就是验这两条声明。
    位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan、第 4 个 1.0(命中 log(1-p)→-inf 的端点)。
    元素数 < 5 不注入。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return pred
    a = np.array(pred, copy=True)
    if a.size < 5:
        return pred
    f = a.reshape(-1)
    f[0], f[1], f[2], f[3] = np.inf, -np.inf, np.nan, 1.0
    return a


def _make_onehot(target, testcase_name=""):
    """`_onehot` 档的定点构造: 把 target 整成**严格 one-hot**(每行恰一个 1, 其余 0)。

    为什么需要钩子: README 声明 target 应为 one-hot, 但本算子不校验取值 —— 非 one-hot 时
    其数值会作为权重直接参与行内求和(结果不再是 Focal Loss 的定义值)。CSV 的
    input_data_ranges 只能给区间, 表达不了"每行恰一个 1", 所以靠随机值域永远测不到
    "严格 one-hot"这一支, 而那恰恰是算子对外承诺的正常输入形态。

    构造: 第 i 行的热点列取 i % C, 其余置 0。不随机, 便于复现。
    """
    if "_onehot" not in (testcase_name or ""):
        return target
    t = np.asarray(target).copy()
    if t.ndim < 1 or t.shape[-1] < 1:
        return target
    c = t.shape[-1]
    flat = t.reshape(-1, c)
    flat[:] = 0
    for row in range(flat.shape[0]):
        flat[row, row % c] = 1
    return t


class SoftmaxFocalLossKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray, 返 numpy.ndarray。

    参数名取自 op_host/softmax_focal_loss_def.cpp: pred / target / weight。
    """

    def golden(*inputs, **kwargs):
        t = [
            None if a is None else torch.from_numpy(np.ascontiguousarray(a))
            for a in inputs
        ]
        outs = _compute(*t, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy()
            for i, o in enumerate(outs)
        ]

    def customize_inputs(pred, target, weight=None, **kwargs):
        name = kwargs.get("testcase_name", "")
        return (
            _inject_nonfinite(pred, name),
            _make_onehot(target, name),
            weight,
        )

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路: canndev 老树 ops/built-in/op_api 与新树 ops/ 下均无
#   softmax_focal_loss 的 op_api 实现, 也无 docs/aclnnSoftmaxFocalLoss.md(01 §3.3)。
# 【不存在】e2e(torch) 通路: torch_npu 2.10.0 的 libtorch_npu.so 无 aclnnSoftmaxFocalLoss
#   符号 —— aclnn 本就不存在, torch 无从下发。
# 【不存在】tf / onnx / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter。
