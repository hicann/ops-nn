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

    third_party = {"torch": _Relu6GradCompose}
    tolerance = _TOL_KERNEL


__spec__ = {"relu6_grad": "Relu6GradKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# TensorFlow: 有 framework 的 tf_plugin, 但 TF 不是 TestSpec 的注册通路,
# 如需对标应以 third_party 的 tf vendor 形式补充, 本次未做。
# e2e / ONNX / 融合 pass: 均未交付。
