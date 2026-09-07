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
# 语义已实测坐实(A100 xpu-server, tf 2.21.0): 对 x ∈ {-1, 0, 1e-7, 3, 6-1e-6, 6, 7,
# NaN, +inf, -inf} 与含 inf/NaN 的 dy, tf.raw_ops.Relu6Grad 的输出与本文件 golden
# **逐位相等**——两端点 0/6 都判 0(严格开区间)、NaN 判 0、带内原样透传 dy 的 inf/NaN。
# 这正是 torch 腿要手写拼接的原因(hardtanh_backward 在 NaN 上分叉), tf 侧不存在该分叉。
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


__spec__ = {"relu6_grad": "Relu6GradKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# TensorFlow: 有 framework 的 tf_plugin(OriginOpType "Relu6Grad")。
#   三方标杆(精度/性能)已补: third_party["tf"] = tf.raw_ops.Relu6Grad, 跑 --provider tf。
#   通路连通(ⓐ)未注册: TTK 的 tf 通路是 e2e 前端(api_name 写 TF API), NPU 侧需要
#   Ascend TF adapter(npu_device/tfplugin)才能把 TF 图下沉到本算子; 当前环境
#   (cann-9.2.0)未装该组件, 装不上就跑不出 invoke_path 证据, 故不注册空壳键
#   (规范: __spec__ 注册集合必须等于 01 §3.3 的 ✅ 集合与 invoke_path 的通路取值)。
#   该组件到位前, tf 通路连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证。
# e2e / ONNX / 融合 pass: 均未交付。
