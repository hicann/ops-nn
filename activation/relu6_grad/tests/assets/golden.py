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

import numpy as np
import torch

__golden__ = {"kernel": {"relu6_grad": "relu6_grad_golden"}}


def relu6_grad_golden(gradients, features, **kwargs):
    """
    Kernel golden for relu6_grad.
    All the parameters follow @relu6_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: dx = (0 < x < 6) ? dy : 0 (element-wise, NumPy broadcasting).
    Strict open-interval: x == 0 and x == 6 both yield 0. NaN in x yields 0
    (any comparison with NaN is false). dy is passed through verbatim when
    the mask is true, so dy = NaN/Inf propagates only inside the (0, 6) band.
    """
    dtype = gradients.dtype
    if dtype == np.float32:
        x = features
        dy = gradients
    else:
        # half / bf16 path: lift to fp32 for the intermediate compute to
        # match the kernel's Relu6GradFloatCast template; cast back at the end.
        x = features.astype(np.float32)
        dy = gradients.astype(np.float32)
    mask = (x > np.float32(0.0)) & (x < np.float32(6.0))
    out = np.where(mask, dy, np.float32(0.0)).astype(dtype)
    return out


# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# 上面的 golden 是纯 numpy 公式；这里补 torch 拼接作三方参照，在设备侧跑供 cross_check
# 比对——纯 numpy 参照与被测 kernel 易犯同类错误，会掩盖精度短板。
#
# 【为何不用 torch 的自然对标 API】aten.hardtanh_backward(dy, x, 0, 6) 看似正好对应
# Relu6 的反向，但它的判据是 (x <= 0) | (x >= 6) -> 0，NaN 对两个比较都为假, 于是
# **透传 dy**；而本算子的定义是靠 (x > 0) & (x < 6) 取掩码, NaN 落到 else 分支 -> 0
# （见上面 golden 的语义说明）。两者只在 NaN 输入上分叉, 直接拿 hardtanh_backward
# 当三方会在 NaN 用例上假红, 故按算子定义用 torch 张量运算拼接。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


# 【tf 腿】本算子的 tf_plugin 把 TF 的 Relu6Grad 逐字映射过来
# (framework/relu6_grad_tf_plugin.cpp: OriginOpType("Relu6Grad")), 故 tf 通路的对标标杆
# 就是 TF 自己那个算子——按交付规范"tf 通路 → 对标 tf, 别拿 torch 顶替 tf 语义"。
#
# TensorFlow 的有限值语义与本算子一致；其原生内核实现为
# gradients * cast((features > 0) & (features < 6))。因此 dy 在带外为 NaN/Inf 时，
# TensorFlow 会按 IEEE-754 产生 NaN，而 CANN kernel/golden 的 Select 语义输出 0。
# kernel 的 tf 三方腿用于有限值交叉验证；TensorFlow E2E golden 则严格遵循 TF 公式。
#
# ⚠️ 为何包一层类, 不写成 third_party={"tf": "tf.raw_ops.Relu6Grad"} 的 API 直调:
# 用例 CSV 带 input_formats 时, 服务端会把逐输入 format 并进调用 kwargs
# (executor.py: compose_kwargs["input_formats"]), 直调形式会撞上
# `TypeError: relu6_grad() got an unexpected keyword argument 'input_formats'`
# → 三方腿整批 FAIL、精度判 GOLDEN_FAILURE(实测跑批 0/6, 探针不带 input_formats 时却全绿,
# 所以这个坑只有真实跑批能暴露)。适配类用 **kwargs 吞掉上下文参数即可, 计算仍是 TF 那一个算子。
class _Relu6GradTfCompose:
    def __call__(self, gradients, features, **kwargs):
        import tensorflow as tf

        return [tf.raw_ops.Relu6Grad(gradients=gradients, features=features)]


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return t.to(torch.float32)


class _Relu6GradCompose:
    def __call__(self, gradients, features, **kwargs):
        dy, x = _tp_t(gradients), _tp_t(features)
        mask = (x > 0.0) & (x < 6.0)
        return [torch.where(mask, dy, torch.zeros_like(dy))]


class Relu6GradKernelSpec:
    @staticmethod
    def golden(gradients, features, **kwargs):
        return [relu6_grad_golden(gradients, features, **kwargs)]

    # dict 顺序 = 优先级: 不带 --provider 时取第一个(torch), 供 kernel/geir 通路对标;
    # 验 tf 通路时显式 --provider tf 切到 tf 腿(--provider 只做过滤, 不改本处配置)。
    third_party = {"torch": _Relu6GradCompose, "tf": _Relu6GradTfCompose}
    tolerance = _TOL_KERNEL


def _to_numpy_array(value):
    """Convert a NumPy or framework tensor to an independent host array."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            from ml_dtypes import bfloat16

            return tensor.view(torch.int16).numpy().view(dtype=bfloat16).copy()
        return tensor.numpy().copy()
    if hasattr(value, "numpy"):
        return np.array(value.numpy(), copy=True)
    return np.array(value, copy=True)


def _promote_reference_array(array):
    """Promote floating inputs so cross-check uses an independent CPU truth."""
    target_dtype = {
        "float16": np.float32,
        "bfloat16": np.float32,
        "float32": np.float64,
    }.get(array.dtype.name)
    return array.astype(target_dtype) if target_dtype is not None else array


def tensorflow_relu6_grad_golden(gradients, features, name=None, **kwargs):
    """CPU golden for ``tf.raw_ops.Relu6Grad``."""
    del name, kwargs
    dy = _promote_reference_array(_to_numpy_array(gradients))
    x = _promote_reference_array(_to_numpy_array(features))
    mask = ((x > 0) & (x < 6)).astype(dy.dtype, copy=False)
    with np.errstate(invalid="ignore"):
        return [dy * mask]


class Relu6GradTensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    golden = staticmethod(tensorflow_relu6_grad_golden)
    third_party = {"tf": _Relu6GradTfCompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "relu6_grad": "Relu6GradKernelSpec",
    "tf.raw_ops.Relu6Grad": "Relu6GradTensorFlowSpec",
}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# TensorFlow E2E: framework 中已注册 OriginOpType "Relu6Grad"；TTK 用
# tf.raw_ops.Relu6Grad 作为 CSV api_name，通过 Relu6GradTensorFlowSpec
# 取得 golden、third_party 和 tolerance。
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# Torch E2E / ONNX / 融合 pass: 均未交付。
