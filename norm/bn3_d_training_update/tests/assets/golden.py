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
"""bn3d_training_update golden(竞品算子拼接实现,torch 张量算子)+ 多通路 TestSpec.

- 上库件:`__golden__` / `__input__`(TTK kernel 模式,签名照 def.cpp)。
- 验证件:`__spec__` —— kernel 与 geir 共用同一注册键与 Spec 类;aclnn/e2e 不存在
  (见文件末尾)。
- 计算核只写一遍:`_compute()` torch 进出;KernelSpec.golden 只做 numpy↔torch 容器转换。
- 三方腿(`_Bn3dThirdPartyCompose`):乘倒数求矩、先减后乘的方差、
  `(x−bm)·mult+offset`、展开式 EMA。

y = (x − batch_mean)·mult + offset(Sub → Mul → Add 三次舍入),中间精度 fp32,
y 回 cast 到 x 的 dtype(fp16/bf16/fp32)。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

_FORMAT_CH_AXIS = {"NCHW": 1, "NCDHW": 1, "NHWC": 3, "NDHWC": 4}

# ---------------------------------------------------------------------------
# C 轴置于 dim1 的规范布局,三方腿与计算核 `_compute` 不共享实现。
# ---------------------------------------------------------------------------
_C1_PERM = {
    "NCHW": (0, 1, 2, 3),
    "NCDHW": (0, 1, 2, 3, 4),
    "NHWC": (0, 3, 1, 2),
    "NDHWC": (0, 4, 1, 2, 3),
}


def _inv_perm(perm):
    """permute 的逆置换(把规范布局还原回下发布局)。"""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


# ---------------------------------------------------------------------------
# numpy(ml_dtypes bf16)↔ torch 位保真转换:经 uint16 view 搬运,无舍入。
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


__golden__ = {"kernel": {"bn3_d_training_update": "bn3d_training_update_golden"}}
__input__ = {"kernel": {"bn3_d_training_update": "bn3_d_training_update_input_gen"}}

# kernel 与 geir 共用同一个注册键(算子蛇形名)与类;aclnn/e2e 不存在(见文件末尾注释)。
__spec__ = {"bn3_d_training_update": "Bn3dKernelSpec"}


# ---------------------------------------------------------------------------
# 工具:channel 轴解析 —— 纯 format 驱动(TTK 恒有 input_formats),
# 歧义 shape 由 format 唯一确定;拿不到 format 直接报错。
# ---------------------------------------------------------------------------
def _resolve_ch_axis(x_shape, C, kwargs):
    fmt = kwargs.get("input_formats", None)
    if fmt is not None and len(fmt) > 0 and fmt[0] in _FORMAT_CH_AXIS:
        return _FORMAT_CH_AXIS[fmt[0]]
    raise ValueError(
        "no input format for channel axis (need input_formats in kwargs): "
        "x.shape=%s C=%s" % (tuple(x_shape), C)
    )


# ---------------------------------------------------------------------------
# 输入生成器:sum/square_sum 由 x 重算(numpy),与前驱 BN3DTrainingReduce 语义一致。
# ---------------------------------------------------------------------------
def bn3_d_training_update_input_gen(*input_arrays, **kwargs):
    if len(input_arrays) < 7:
        return list(input_arrays)
    x = np.asarray(input_arrays[0])
    C = input_arrays[1].shape[0]
    ch = _resolve_ch_axis(x.shape, C, kwargs)
    x_f32 = x.astype(np.float32)
    axes = tuple(i for i in range(x.ndim) if i != ch)
    sum_arr = x_f32.sum(axis=axes).astype(np.float32)
    sq_arr = (x_f32 * x_f32).sum(axis=axes).astype(np.float32)
    result = list(input_arrays)
    result[1] = sum_arr
    result[2] = sq_arr
    return result


# ---------------------------------------------------------------------------
# 计算核(全算子只写这一份,torch 进出)——算法语义见下。
# ---------------------------------------------------------------------------
def _compute(
    x, sum_t, square_sum, scale, offset, mean, variance, factor, epsilon, ch_axis
):
    """BN3DTrainingUpdate 语义;x 可为 fp16/fp32/bf16,统计量 fp32。"""
    x_dt = x.dtype
    x_w = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    C = sum_t.shape[0]
    numel = x.numel()
    # 空 tensor 守卫:C=0 时 num 无定义(python 整除零直接抛 ZeroDivisionError)。
    num = (numel // C) if C > 0 else 0

    # 与算子实现同序:乘倒数 num_rec(非除法,fp32 下差 1ULP)。
    # num=0 时零统计:bm/bv=0、mean_out=mean·(1-f)、var_out=var·(1-f)。
    num_rec = (1.0 / num) if num > 0 else 0.0
    batch_mean = sum_t * num_rec
    save_variance = square_sum * num_rec - batch_mean * batch_mean

    inv_std = 1.0 / torch.sqrt(save_variance + epsilon)
    multiplier = scale * inv_std

    stat_shape = [1] * x.dim()
    stat_shape[ch_axis] = C
    y = (x_w - batch_mean.reshape(stat_shape)) * multiplier.reshape(
        stat_shape
    ) + offset.reshape(stat_shape)
    y = y.to(x_dt)

    # Bessel 修正:num==1 时分母为 0 → 无偏 batch variance 置 0(running variance 不更新)。
    scaler = 0.0 if num == 1 else float(num) / float(num - 1)
    unbiased_batch_var = save_variance * scaler

    mean_out = mean * (1.0 - factor) + batch_mean * factor
    variance_out = variance * (1.0 - factor) + unbiased_batch_var * factor

    return [y, mean_out, variance_out, batch_mean, save_variance]


# ---------------------------------------------------------------------------
# kernel/geir Spec:numpy 进出,golden 只做容器转换后调 _compute。
# ---------------------------------------------------------------------------


def _pin_eager_fp32_numerics():
    """把 `torch.compile` 钉在「eager fp32 逐步舍入」精度档(禁 FMA 收缩)。

    Inductor 的 Triton 后端默认把 `a * b ± c` 融合成单次舍入的 FMA,而内核刻意走
    三步非融合 `Sub + Mul + Add`;置 `emulate_precision_casts=True` 即关闭 Triton 的
    `contract` 标志,使竞品与内核同精度档。只关 fp 融合,融合图 / 单 kernel 形态不变。
    该配置参与 Inductor 缓存键,不会复用带 FMA 的旧产物;且必须在进程内首次编译前
    置位(先编译过再置位不会重编)。

    返回旋钮是否已生效;不可用 → False,由 `__call__` 直接报错。
    """
    try:
        from torch._inductor import config as _inductor_config
    except Exception:
        return False
    try:
        if not getattr(_inductor_config, "emulate_precision_casts", False):
            _inductor_config.emulate_precision_casts = True
    except Exception:
        return False
    return bool(getattr(_inductor_config, "emulate_precision_casts", False))


def _fp32_pin_unavailable_error():
    """`_pin_eager_fp32_numerics()` 失败时的报错(原因 + 实际环境 + 处置)。"""
    try:
        from torch._inductor import config as _inductor_config

        where = "torch._inductor.config 可导入"
        has_attr = hasattr(_inductor_config, "emulate_precision_casts")
    except Exception as exc:
        where = "torch._inductor.config 不可导入(%s: %s)" % (
            type(exc).__name__,
            exc,
        )
        has_attr = False
    return RuntimeError(
        "三方腿需要 torch._inductor.config.emulate_precision_casts=True"
        "(禁 Inductor/Triton 的 FMA 收缩,与内核三步非融合 Sub/Mul/Add 同精度档),"
        "当前环境不可用。实际环境:torch.__version__=%s;%s;"
        "emulate_precision_casts 属性存在=%s。"
        "处置:升级 torch >=2.12,或在进程内首次 torch.compile 前显式置位。"
        % (torch.__version__, where, has_attr)
    )


class _Bn3dThirdPartyCompose:
    """三方腿:竞品(torch)独立实现(框架固定 torch)。

    算术步骤与内核同序:
      1. 逐通道矩:乘倒数 `bm = Σx·num_rec`,先减后乘 `(Σx² − Σx·bm)·num_rec`;
      2. 元素级仿射:`(x − bm)·mult + offset`(Sub → Mul → Add),`mult = scale/sqrt(var+eps)`;
      3. running 统计:展开式 `bm·f + mean·(1−f)`(非 lerp)。
    入口按内核加宽行为 cast(f16/bf16 → f32),出口窄回输入 dtype(与 CAST_RINT
    round-half-even 同口径);x 仅 f16/f32/bf16,无整型分支。

    形态:同一 compose 同时驱动精度腿与性能腿(TTK 不透传腿别)——保留
    `torch.compile(mode="reduce-overhead")` 融合形态,融合图内禁 FMA 收缩,
    逐步舍入与 eager fp32 / 内核同档。环境不满足或 `torch.compile` 失败直接报错。
    """

    def __init__(self, **kwargs):
        self.factor = float(kwargs.get("factor", 0.1))
        self.epsilon = float(kwargs.get("epsilon", 1.0e-5))
        # 输入 format(内嵌在 X-Input-Schema 传入):有则 format-aware 定通道轴
        self.input_formats = kwargs.get("input_formats") or None

    def _perm(self, x):
        """format 驱动定「C 轴在 dim1」的重排元组;拿不到直接报错。
        `input_formats` 的序列与裸字符串形态都收。
        """
        fmts = self.input_formats
        if not fmts:
            raise ValueError(
                "compose: no input format for channel axis (need input_formats): "
                "x.shape=%s" % (tuple(x.shape),)
            )
        fmt = str(fmts[0] if isinstance(fmts, (list, tuple)) else fmts).upper()
        perm = _C1_PERM.get(fmt)
        if perm is None or len(perm) != x.dim():
            raise ValueError(
                "compose: unsupported format/rank: format=%r x.shape=%s"
                % (fmt, tuple(x.shape))
            )
        return perm

    def _update_eager(self, x, sum_t, square_sum, scale, offset, mean, variance):
        out_dt = x.dtype
        perm = self._perm(x)
        identity = perm == tuple(range(x.dim()))
        C = int(sum_t.shape[0])
        numel = x.numel()
        # 空 batch 契约(与契约一致):num=0 ⇒ 零统计量,不抛除零。
        num = (numel // C) if C > 0 else 0

        # 入口:按 NPU 的加宽行为同步 cast(内核 Cast(CAST_NONE) 只作用于 f16/bf16)。
        if out_dt in (torch.float16, torch.bfloat16):
            xn = x.to(torch.float32)
        else:
            xn = x

        # 乘倒数(num_rec)、先减后乘的二阶中心矩、scale/sqrt(var+eps)。
        if num > 0:
            num_rec = 1.0 / num
            batch_mean = sum_t * num_rec
            save_variance = (square_sum - sum_t * batch_mean) * num_rec
        else:
            batch_mean = torch.zeros_like(sum_t)
            save_variance = torch.zeros_like(square_sum)

        if numel > 0:
            x_c1 = xn if identity else xn.permute(*perm)
            # elementwise,与布局无关;Sub → Mul → Add。
            stat_shape = [1] * x_c1.dim()
            stat_shape[1] = C
            multiplier = scale / torch.sqrt(save_variance + self.epsilon)
            y = (x_c1 - batch_mean.reshape(stat_shape)) * multiplier.reshape(
                stat_shape
            ) + offset.reshape(stat_shape)
            if not identity:
                y = y.permute(*_inv_perm(perm))
        else:
            # C==0:infershape 只拷贝 shape、tiling 支持 ⇒ 产出空 y。
            y = torch.empty_like(xn)
        # 出口:窄回下发输入 dtype,与内核 CAST_RINT(round-half-even)同口径。
        y = y.to(out_dt)

        # Bessel 修正:num==1 时分母为 0 → 无偏 batch variance 置 0(running variance 不更新)。
        scaler = 0.0 if num == 1 else float(num) / float(num - 1)
        unbiased_batch_var = save_variance * scaler
        # running 统计:展开式 bm·f + mean·(1−f)(非 lerp)。
        mean_out = batch_mean * self.factor + mean * (1.0 - self.factor)
        variance_out = unbiased_batch_var * self.factor + variance * (1.0 - self.factor)
        return [y, mean_out, variance_out, batch_mean, save_variance]

    def __call__(self, x, sum, square_sum, scale, offset, mean, variance):
        args = (x, sum, square_sum, scale, offset, mean, variance)
        # 融合形态(性能腿)+ 禁 FMA 收缩(精度腿,同内核三步非融合)。环境不满足直接报错。
        if not _pin_eager_fp32_numerics():
            raise _fp32_pin_unavailable_error()
        try:
            fn = torch.compile(self._update_eager, mode="reduce-overhead")
            return fn(*args)
        except ValueError:
            # `_perm` 的契约校验原样抛出。
            raise
        except Exception as exc:
            # 编译/首执行失败:包装成 RuntimeError 并保留异常链。
            raise RuntimeError(
                "三方腿 torch.compile(mode='reduce-overhead') 编译或首执行失败"
                "(%s: %s);torch.__version__=%s"
                % (type(exc).__name__, exc, torch.__version__)
            ) from exc


class Bn3dKernelSpec:
    @staticmethod
    def _attr(kwargs, name, default):
        v = kwargs.get(name)
        if v is None:
            attrs = kwargs.get("attributes")
            if isinstance(attrs, dict):
                v = attrs.get(name)
        return default if v is None else v

    @staticmethod
    def golden(x, sum, square_sum, scale, offset, mean, variance, **kwargs):
        factor = float(Bn3dKernelSpec._attr(kwargs, "factor", 0.1))
        epsilon = float(Bn3dKernelSpec._attr(kwargs, "epsilon", 1.0e-5))
        ch = _resolve_ch_axis(np.asarray(x).shape, np.asarray(sum).shape[0], kwargs)
        outs = _compute(
            _to_torch(x),
            _to_torch(np.asarray(sum)),
            _to_torch(np.asarray(square_sum)),
            _to_torch(np.asarray(scale)),
            _to_torch(np.asarray(offset)),
            _to_torch(np.asarray(mean)),
            _to_torch(np.asarray(variance)),
            factor,
            epsilon,
            ch,
        )
        return [_to_numpy(o) for o in outs]

    # 三方标杆(远端 GPU 执行):竞品原生原语拼接的独立实现(见类 docstring);
    # torch.compile 融合为最优形态,融合图内禁 FMA 收缩(`_pin_eager_fp32_numerics`)。
    third_party = {"torch": _Bn3dThirdPartyCompose}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# ---------------------------------------------------------------------------
# 上库件老入口(签名照 def.cpp):numpy 进出,指向同一计算核,供 TTK kernel 模式直调。
# ---------------------------------------------------------------------------
def bn3d_training_update_golden(
    x,
    sum,
    square_sum,
    scale,
    offset,
    mean,
    variance,
    factor=0.1,
    epsilon=1.0e-5,
    **kwargs,
):
    return Bn3dKernelSpec.golden(
        x,
        sum,
        square_sum,
        scale,
        offset,
        mean,
        variance,
        factor=factor,
        epsilon=epsilon,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 三方 compose(远端 GPU 上执行):属性喂 __init__、输入喂 __call__,参数名与 proto
# REG_OP 注册名逐字一致。输出 cast 到 NPU 输出 dtype(y 回 x dtype;统计量 fp32)。
# `_compute` 为 golden 腿独立参考;`_Bn3dThirdPartyCompose` 按被测实现的运算序列拼接,
# 避免等价写法凭空更准而把实现差异误判成内核缺陷。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 通路支持面:
#   kernel = ✅(op_kernel/arch35 实现)
#   geir   = ✅(op_graph/bn3_d_training_update_proto.h REG_OP)
#   aclnn  = ❌(op_host/CMakeLists.txt ACLNNTYPE aclnn_exclude;无 op_api 实现)
#   e2e    = ❌(canndev 无本算子 aclnn → torch_npu 无绑定可派发)
# aclnn/e2e 不存在 → 不注册对应 __spec__ 条目(不留空壳,TTK 会真的加载空实现)。
# ---------------------------------------------------------------------------
