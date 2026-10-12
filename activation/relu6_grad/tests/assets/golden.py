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
    import tensorflow as tf

    with tf.device("/CPU:0"):
        dy = tf.convert_to_tensor(np.asarray(gradients))
        x = tf.convert_to_tensor(np.asarray(features))
        output_dtype = dy.dtype
        promote = kwargs.get("golden_mode") == "Promote"
        if not promote and output_dtype in (tf.float16, tf.bfloat16):
            dy, x = tf.cast(dy, tf.float32), tf.cast(x, tf.float32)
        mask = tf.logical_and(x > 0.0, x < 6.0)
        # Kernel uses Select.  Native TF Relu6Grad multiplies by the mask and
        # produces NaN for out-of-band NaN/Inf gradients; keep the paths distinct.
        output = tf.where(mask, dy, tf.zeros_like(dy))
        if not promote:
            output = tf.cast(output, output_dtype)
        return np.asarray(output.numpy())


# ----------------------------------------------------------------------------
# Use strict comparisons instead of aten.hardtanh_backward: the latter passes
# dy through when features is NaN, unlike both the kernel and TensorFlow masks.
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


class _Relu6GradCompose:
    """Torch GPU reference for the kernel's select semantics."""

    def __call__(self, gradients, features, **kwargs):
        del kwargs
        import torch

        output_dtype = gradients.dtype
        dy, x = gradients, features
        if output_dtype in (torch.float16, torch.bfloat16):
            dy, x = dy.to(torch.float32), x.to(torch.float32)
        mask = (x > 0.0) & (x < 6.0)
        return [torch.where(mask, dy, torch.zeros_like(dy)).to(output_dtype)]


class _Relu6GradE2ECompose:
    """Torch GPU reference for native TF's multiply-by-mask semantics."""

    def __call__(self, gradients, features, **kwargs):
        del kwargs
        import torch

        output_dtype = gradients.dtype
        dy, x = gradients, features
        if output_dtype in (torch.float16, torch.bfloat16):
            dy, x = dy.to(torch.float32), x.to(torch.float32)
        mask = ((x > 0.0) & (x < 6.0)).to(dy.dtype)
        return [(dy * mask).to(output_dtype)]


class Relu6GradKernelSpec:
    @staticmethod
    def golden(gradients, features, **kwargs):
        return [relu6_grad_golden(gradients, features, **kwargs)]

    third_party = {"torch": _Relu6GradCompose}
    tolerance = _TOL_KERNEL


class Relu6GradTensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    # Let TTK promote raw inputs and execute tf.raw_ops.Relu6Grad on CPU.
    # A custom E2E golden would bypass that framework-owned promotion.
    third_party = {"torch": _Relu6GradE2ECompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "relu6_grad": "Relu6GradKernelSpec",
    "tf.raw_ops.Relu6Grad": "Relu6GradTensorFlowSpec",
}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# TensorFlow E2E: framework 中已注册 OriginOpType "Relu6Grad"；TTK 用
# tf.raw_ops.Relu6Grad 作为 CSV api_name，通过 Relu6GradTensorFlowSpec
# 取得 third_party 和 tolerance，CPU golden 复用框架默认 TensorFlow 通路。
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# Torch E2E / ONNX / 融合 pass: 均未交付。
