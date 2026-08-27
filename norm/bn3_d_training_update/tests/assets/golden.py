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

- 上库件(随算子工程):`__golden__` / `__input__`(TTK kernel 模式,签名照 def.cpp)。
- 验证件(同文件):`__spec__` —— kernel 与 geir 共用同一注册键与 Spec 类;aclnn/e2e 不存在
  (依据见文件末尾)。
- 计算核只写一遍:`_compute()` torch 进出;KernelSpec.golden 只做 numpy↔torch 容器转换。

golden 采用竞品算子拼接(torch 张量算子:mul/sub/div/sqrt 等),不手写 numpy 纯公式。
计算顺序与算子契约(01/03)一致:FMA 形式 y = x*mult + addend(mult/addend 由逐通道
统计量预算而来),中间精度 fp32,y 回 cast 到 x 的 dtype(fp16/bf16/fp32)。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

_FORMAT_CH_AXIS = {"NCHW": 1, "NCDHW": 1, "NHWC": 3, "NDHWC": 4}


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


__golden__ = {"kernel": {"bn3_d_training_update": "bn3d_training_update_golden"}}
__input__ = {"kernel": {"bn3_d_training_update": "bn3_d_training_update_input_gen"}}

# kernel 与 geir 共用同一个注册键(算子蛇形名)与类;aclnn/e2e 不存在(见文件末尾注释)。
__spec__ = {"bn3_d_training_update": "Bn3dKernelSpec"}


# ---------------------------------------------------------------------------
# 工具:channel 轴解析 —— 纯 format 驱动
# format 在 TTK 场景恒有(input_formats 经 X-Input-Schema 内嵌传给 golden/compose),
# NCHW/NHWC 且空间维等于 C 的歧义 shape 由 format 唯一确定;拿不到 format 直接报错。
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
# 输入生成器:sum/square_sum 由 x 重算,保证与前驱 BN3DTrainingReduce 的语义一致。
# (输入生成属数据准备,非 golden 计算,保持 numpy。)
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
# 计算核(全算子只写这一份,torch 进出)——算法语义照 01 §2 / 03 spec。
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

    # 与算子实现同序:乘倒数(num_rec),非除法(除法与倒数乘在 fp32 下差 1ULP)。
    # 空 batch 契约(tiling num_rec=0 同款):num=0 时零统计——bm/bv=0、
    # mean_out=mean·(1-f)、var_out=var·(1-f)。竞品佐证:torch CUDA native 对空
    # 归约域 save_mean=0(非 nan),CPU/拼接原版则直接拒绝/崩溃,无 nan 语义。
    num_rec = (1.0 / num) if num > 0 else 0.0
    batch_mean = sum_t * num_rec
    save_variance = square_sum * num_rec - batch_mean * batch_mean

    inv_std = 1.0 / torch.sqrt(save_variance + epsilon)
    multiplier = scale * inv_std
    addend = offset - multiplier * batch_mean

    stat_shape = [1] * x.dim()
    stat_shape[ch_axis] = C
    y = x_w * multiplier.reshape(stat_shape) + addend.reshape(stat_shape)
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


class _Bn3dThirdPartyCompose:
    def __init__(self, **kwargs):
        self.factor = float(kwargs.get("factor", 0.1))
        self.epsilon = float(kwargs.get("epsilon", 1.0e-5))
        # 输入 format(内嵌在 X-Input-Schema 传入):有则 format-aware 定通道轴
        self.input_formats = kwargs.get("input_formats") or None

    def _update_eager(self, x, sum_t, square_sum, scale, offset, mean, variance):
        x_dt = x.dtype
        x_f32 = x.float()
        C = sum_t.shape[0]
        numel = x.numel()
        num = numel // C

        num_rec = (1.0 / num) if num > 0 else 0.0  # 空 batch 契约,与 _compute 同款
        batch_mean = sum_t * num_rec
        save_variance = square_sum * num_rec - batch_mean * batch_mean
        inv_std = 1.0 / torch.sqrt(save_variance + self.epsilon)
        multiplier = scale * inv_std
        addend = offset - multiplier * batch_mean

        # 通道轴:纯 format 驱动(与 _resolve_ch_axis 同源);无 format 直接报错,不猜轴。
        ch = None
        if self.input_formats:
            fmt = str(self.input_formats[0]).upper()
            ch = _FORMAT_CH_AXIS.get(fmt)
        if ch is None:
            raise ValueError(
                "compose: no input format for channel axis (need input_formats): "
                "x.shape=%s C=%s" % (tuple(x.shape), C)
            )
        view = [1] * x.dim()
        view[ch] = C
        y = x_f32 * multiplier.view(view) + addend.view(view)

        scaler = 0.0 if num == 1 else float(num) / float(num - 1)
        unbiased_batch_var = save_variance * scaler
        mean_out = mean * (1.0 - self.factor) + batch_mean * self.factor
        variance_out = variance * (1.0 - self.factor) + unbiased_batch_var * self.factor
        return [y.to(x_dt), mean_out, variance_out, batch_mean, save_variance]

    def __call__(self, x, sum, square_sum, scale, offset, mean, variance):
        try:
            fn = torch.compile(self._update_eager, mode="reduce-overhead")
            return fn(x, sum, square_sum, scale, offset, mean, variance)
        except Exception:
            return self._update_eager(x, sum, square_sum, scale, offset, mean, variance)


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

    # 三方标杆(远端 GPU 执行):与算子实现同算法(FMA 形式),torch.compile 融合为最优形态,
    # 编译失败回落 eager。
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
# 三方 compose(远端 GPU 上执行):类形式,属性喂 __init__、输入喂 __call__;
# 参数名与 proto REG_OP 注册名逐字一致(x, sum, square_sum, scale, offset,
# mean, variance, factor, epsilon)。torch.compile 包一层拿最优形态,失败回落 eager。
# 输出必须 cast 到 NPU 输出 dtype(y 回 x dtype;统计量 fp32),保证 cross_check 同精度对等。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 通路支持面(照 01_requirement.md §3.3):
#   kernel = ✅(op_kernel/arch35 实现)
#   geir   = ✅(op_graph/bn3_d_training_update_proto.h REG_OP)
#   aclnn  = ❌(op_host/CMakeLists.txt ACLNNTYPE aclnn_exclude;无 op_api 实现)
#   e2e    = ❌(canndev 无本算子 aclnn → torch_npu 无绑定可派发)
# aclnn/e2e 不存在 → 不注册对应 __spec__ 条目(不留空壳,TTK 会真的加载空实现)。
# ---------------------------------------------------------------------------
