#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""bn3_d_training_reduce_grad golden(竞品算子拼接实现,torch 张量算子)+ kernel/geir TestSpec.

- 上库件(随算子工程):`__golden__`(TTK kernel 模式,签名照 def.cpp)。
- 验证件(同文件):`__spec__` —— kernel 与 geir 共用同一注册键与 Spec 类;aclnn/e2e 不存在
  (依据见文件末尾)。
- 计算核只写一遍:`_compute()` torch 进出;KernelSpec.golden 只做 numpy↔torch 容器转换;
  三方 compose 也指向同一计算核(外加通道轴消歧与 torch.compile 融合)。

golden 采用竞品算子拼接(torch 张量算子 sub/mul/div/sqrt),不手写 numpy 纯公式(红线 R3:
纯公式常与被测实现犯同一个理解错误,失去交叉验证意义)。

运算序列与 arch35 内核 5 条 VF 链逐步对齐(bn3d_training_reduce_grad_kernel.h):
    s   = sqrt(bv + eps)             ChainS : Adds+Sqrt(epsilon_guard)
    t_a = (x - bm) * ds * inv_num    ChainA1: Sub+Mul+Muls
    t1  = grads - t_a / s            ChainA2: Div+Sub(÷s 在括号内、先于 grads−)
    t2  = t1 - do * inv_num          ChainB : Muls+Sub
    y   = (t2 * sc) / s              ChainC : Mul+Div
f16/bf16 输入升 f32 参与全部中间运算,结果回落原 dtype(kernel CAST_RINT=RN-even,
torch .to(float16/bfloat16) 同口径);num = 除通道轴外所有维的乘积(NCDHW/NDHWC 一致)。

精度档位契约:
- tolerance 配 cross_check 时框架自动把输入抬一档(f16/bf16→f32、f32→f64)喂给 golden,
  golden 只许向上兜底、绝不向下砍 —— _work_dtype 只把 f16/bf16 抬到 f32,f32/f64 照单全收;
- 当前阶段本机跑 NPU,判定用 stat_rel_err(阈值=spec numerical_tolerance.rtol);
  三方 cross_check 阶段把 tolerance 换成 {"standard": "cross_check", "level": "L1"}
  (需 XPU endpoint/SSH 隧道,家族先例 gpu-20102),golden 计算核无需改动。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

# 通道轴:按声明的 format 严格判定

_FORMAT_CH_AXIS = {"NCDHW": 1, "NDHWC": 4}

# CANN proto 口径默认 0.0001(非 PyTorch 的 1e-5,由插件层显式映射)
_DEFAULT_EPS = 1.0e-4


# ---------------------------------------------------------------------------
# numpy(ml_dtypes bf16)↔ torch 位保真转换:torch.from_numpy 不认 ml_dtypes
# bfloat16 ndarray,经 uint16 view 逐位搬运,无任何舍入。
# ---------------------------------------------------------------------------
def _to_torch(a):
    a = np.asarray(a)
    if a.dtype.name == "bfloat16" and _bf16 is not None:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _to_numpy(t):
    if t.dtype == torch.bfloat16 and _bf16 is not None:
        return t.view(torch.uint16).numpy().view(_bf16)
    return t.numpy()


__golden__ = {
    "kernel": {"bn3_d_training_reduce_grad": "bn3d_training_reduce_grad_golden"}
}

# kernel 与 geir 共用同一个注册键(算子蛇形名,与 def.cpp opInterface.value 一致)与类;
# aclnn/e2e 不存在(见文件末尾注释)。
__spec__ = {"bn3_d_training_reduce_grad": "Bn3dReduceGradKernelSpec"}


# ---------------------------------------------------------------------------
# 工具:channel 轴解析 —— 以 kwargs['input_formats'] 声明为准(TTK 下发的键名,
# 旧版 golden 误用 'tensor_formats' 收不到,恒走抛错分支)。
# ---------------------------------------------------------------------------
def _resolve_ch_axis(shape, c, kwargs):
    if len(shape) != 5:
        # 检查器用通用 2D/低维输入探测 golden 时, 本算子语义要求 5D 才会到这里;
        # 抛清晰错误以区分「探测输入非法」与「format 不支持」。
        raise ValueError(
            "BN3DTrainingReduceGrad grads 必须 5D(探测输入 %r 非法); "
            "golden 只处理合法 5D 用例" % (tuple(shape),)
        )
    fmts = kwargs.get("input_formats")
    fmt = None
    if fmts:
        f0 = fmts[0] if isinstance(fmts, (list, tuple)) else fmts
        fmt = str(f0).upper()
    if fmt in _FORMAT_CH_AXIS:
        return _FORMAT_CH_AXIS[fmt]
    raise ValueError(
        "unsupported tensor format for 5D grads: %r (must be NCDHW/NDHWC, "
        "shape %s, C=%s)" % (fmt, tuple(shape), c)
    )


def _work_dtype(*tensors):
    """计算档位:只向上兜底 —— f16/bf16 抬 f32;任一输入是 f64(Promote 抬档)则全 f64。"""
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


# ---------------------------------------------------------------------------
# 计算核(全算子只写这一份,torch 进出)——运算序列照 arch35 内核 5 条 VF 链。
# ---------------------------------------------------------------------------
def _compute(
    grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, epsilon, ch
):
    """BN3DTrainingReduceGrad 语义;grads/x 可为 fp16/fp32/bf16,5 个参数张量 fp32。

    y = (grads - ds*(x-bm)/(num*s) - do/num) * sc/s, s = sqrt(bv + epsilon);
    num = 除通道轴(ch)外所有维的乘积;返回 [y](回落 grads.dtype)。
    """
    out_dt = grads.dtype  # 回落目标:Promote 下=抬档精度(近真值),原生=原 dtype
    work = _work_dtype(
        grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    )
    g = grads.to(work)
    xx = x.to(work)
    ds = diff_scale.to(work)
    do = diff_offset.to(work)
    sc = scale.to(work)
    bm = batch_mean.to(work)
    bv = batch_variance.to(work)

    c = int(ds.shape[0])
    view = [1] * grads.dim()
    view[ch] = c
    num = 1
    for d, s_ in enumerate(grads.shape):
        if d != ch:
            num *= int(s_)
    inv_num = 1.0 / num  # python 标量:·inv_num 替代 ÷num,与 kernel Muls 同口径

    s = torch.sqrt(bv + float(epsilon)).reshape(view)  # ChainS
    t_a = (xx - bm.reshape(view)) * ds.reshape(view) * inv_num  # ChainA1
    t1 = g - t_a / s  # ChainA2
    t2 = t1 - do.reshape(view) * inv_num  # ChainB
    y = (t2 * sc.reshape(view)) / s  # ChainC
    return [y.to(out_dt)]


# ---------------------------------------------------------------------------
# 三方 compose(远端 XPU/GPU server 上执行):属性喂 __init__、输入喂 __call__;
# 参数名与 def.cpp / proto REG_OP 注册名逐字一致(grads, x, diff_scale,
# diff_offset, scale, batch_mean, batch_variance, epsilon)。torch.compile 包一层
# 拿最优形态(竞品必须开融合,分解实现会虚高 G/N),失败回落 eager。
# 输出必须 cast 回 NPU 输出 dtype(y 回 grads dtype),保证三方同精度对等。
#
# 通道轴消歧:server 侧只绑得到 inputs∪attrs(拿不到 input_formats),按 shape 判
# dim1=C ∧ dim4≠C → NCDHW;dim4=C ∧ dim1≠C → NDHWC;两侧同 C 时默认 NCDHW ——
# 三方用例集避免取 dim1==dim4==C 的歧义 shape 即可完全规避。
# 注: 必须定义在 Spec 之前,供 third_party 以类对象引用(xpu-server 只收
# module.attr 点路径,字符串类名会被拒 400 —— 实测教训)。
# ---------------------------------------------------------------------------
class _Bn3dReduceGradCompose:
    def __init__(self, **kwargs):
        self.epsilon = float(kwargs.get("epsilon", _DEFAULT_EPS))
        # TTK 新功能: input_formats 经 X-Input-Schema 内嵌传给 compose __init__
        # (ops-test-kit executor.py compose_kwargs["input_formats"])。
        fmts = kwargs.get("input_formats") or None
        self.input_formats = str(fmts[0]).upper() if fmts else None

    def _rg_eager(
        self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    ):
        # 纯 format 驱动判通道轴(NCDHW→dim1 / NDHWC→dim4),【无 shape 启发式】;
        # 拿不到 format 直接报错, 不做形状猜测(全维度相等歧义由 format 唯一确定)。
        if self.input_formats == "NCDHW":
            ch = 1
        elif self.input_formats == "NDHWC":
            ch = 4
        else:
            raise ValueError(
                "compose: no input format for channel axis (need input_formats): "
                "input_formats=%r" % (self.input_formats,)
            )
        return _compute(
            grads,
            x,
            diff_scale,
            diff_offset,
            scale,
            batch_mean,
            batch_variance,
            self.epsilon,
            ch,
        )

    def __call__(
        self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
    ):
        try:
            fn = torch.compile(self._rg_eager, mode="reduce-overhead")
            return fn(
                grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
            )
        except Exception:
            return self._rg_eager(
                grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance
            )


# ---------------------------------------------------------------------------
# kernel/geir Spec:numpy 进出,golden 只做容器转换后调 _compute。
# ---------------------------------------------------------------------------
class Bn3dReduceGradKernelSpec:
    @staticmethod
    def _attr(kwargs, name, default):
        v = kwargs.get(name)
        if v is None:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                v = attrs.get(name)
        return default if v is None else v

    @staticmethod
    def golden(
        grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance, **kwargs
    ):
        epsilon = float(Bn3dReduceGradKernelSpec._attr(kwargs, "epsilon", _DEFAULT_EPS))
        ch = _resolve_ch_axis(
            np.asarray(grads).shape, np.asarray(diff_scale).shape[0], kwargs
        )
        outs = _compute(
            _to_torch(grads),
            _to_torch(x),
            torch.from_numpy(np.asarray(diff_scale, dtype=np.float32)),
            torch.from_numpy(np.asarray(diff_offset, dtype=np.float32)),
            torch.from_numpy(np.asarray(scale, dtype=np.float32)),
            torch.from_numpy(np.asarray(batch_mean, dtype=np.float32)),
            torch.from_numpy(np.asarray(batch_variance, dtype=np.float32)),
            epsilon,
            ch,
        )
        return [_to_numpy(o) for o in outs]

    # 三方标杆(远端 XPU server 执行):与内核同算法同序,torch.compile 拿最优形态。
    # 类对象引用(xpu-server 要求 module.attr 点路径;字符串类名会 400 拒收)。
    third_party = {"torch": _Bn3dReduceGradCompose}

    # tolerance 是【未显式传 --compare 时】的默认路由:
    #   - 三方精度: 显式 --compare cross_check(默认即 cross_check, 与 tolerance 一致)
    #   - 泛化/性能/invoke-path(非三方): 必须显式 --compare close(联合 rtol+atol,
    #     避免 stat_rel_err 对近零元素假红), 否则会误路由到 cross_check
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# ---------------------------------------------------------------------------
# 上库件老入口(签名照 def.cpp):numpy 进出,指向同一计算核,供 TTK kernel 模式直调。
# ---------------------------------------------------------------------------
def bn3d_training_reduce_grad_golden(
    grads,
    x,
    diff_scale,
    diff_offset,
    scale,
    batch_mean,
    batch_variance,
    epsilon=_DEFAULT_EPS,
    **kwargs,
):
    return Bn3dReduceGradKernelSpec.golden(
        grads,
        x,
        diff_scale,
        diff_offset,
        scale,
        batch_mean,
        batch_variance,
        epsilon=epsilon,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 通路支持面:
#   kernel = ✅(op_kernel/arch35 实现,RANK=4/8 双 TilingKey 分支)
#   geir   = ✅(op_graph/bn3_d_training_reduce_grad_proto.h REG_OP)
#   aclnn  = ❌(本算子不交付 aclnn 通路——按需求方口径舍弃;旧 golden 的
#              aclnnBN3DTrainingReduceGrad 注册条目与 TestSpec 已删,不留空壳)
#   e2e    = ❌(无 aclnn 交付 → torch_npu 无绑定可派发)
# aclnn/e2e 不存在 → 不注册对应 __spec__ 条目(留空壳 TTK 会真的加载到空实现)。
# ---------------------------------------------------------------------------
