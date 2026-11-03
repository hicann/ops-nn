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
"""BNInferGrad 独立 Golden：Kernel、GEIR。"""

import torch


__spec__ = {"bn_infer_grad": "BNInferGradKernelSpec"}


_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _compute(grads, scale, batch_variance, epsilon, format_x):
    is_torch = isinstance(grads, torch.Tensor)
    if not is_torch:
        from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

        grads = numpy_to_torch_tensor(grads)
        scale = numpy_to_torch_tensor(scale)
        batch_variance = numpy_to_torch_tensor(batch_variance)
    shape_x = grads.shape
    output_dtype = grads.dtype

    if grads.numel() == 0 or scale.numel() == 0 or batch_variance.numel() == 0:
        result = torch.zeros(grads.shape, dtype=output_dtype, device=grads.device)
        return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())

    if len(scale.shape) == 1 and format_x == "NCHW":
        channel_shape = (1, scale.shape[0], *((1,) * (grads.ndim - 2)))
        scale = scale.reshape(channel_shape)
        batch_variance = batch_variance.reshape(channel_shape)

    # 参考实现在 FP64 中计算后再回写输出 dtype：输入取到 ±FLT_MAX 时
    # grads*scale / scale*rsqrt 任一结合顺序在 FP32 中间域都会溢出为 inf，
    # 而真实结果可能仍是有限值（如 scale=-FLT_MAX、variance=FLT_MAX 时
    # 结果为 grads*1.8e19）。FP64 中间域无此溢出，作为参考口径更准确；
    # 输出仍按 dtype 舍入，常规值域与 FP32 计算逐位一致。
    grads_c = grads.to(torch.float64) if grads.is_floating_point() else grads
    scale_c = scale.to(torch.float64) if scale.is_floating_point() else scale
    var_c = (
        batch_variance.to(torch.float64)
        if batch_variance.is_floating_point()
        else batch_variance
    )

    data_add = torch.add(var_c, epsilon)
    data_abs = torch.abs(data_add)
    data_sqrt = torch.sqrt(data_abs)
    # 文档公式：grads * scale / sqrt(|variance + epsilon|)
    scale_broadcast = torch.broadcast_to(scale_c, shape_x)
    result = torch.divide(torch.multiply(grads_c, scale_broadcast), data_sqrt)
    result = result.to(output_dtype)
    return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())


def bn_infer_grad_golden(grads, scale, batch_variance, epsilon=0.0001, **kwargs):
    input_formats = kwargs.get("input_formats") or ()
    format_x = str(input_formats[0]).upper() if input_formats else "NCHW"
    return [_compute(grads, scale, batch_variance, epsilon, format_x)]


class _TorchBNInferGrad:
    """Torch competitor used by three-party precision and performance runs."""

    def __call__(instance, grads, scale, batch_variance, epsilon=0.0001, **kwargs):
        input_formats = kwargs.get("input_formats") or ()
        format_x = str(input_formats[0]).upper() if input_formats else "NCHW"
        return [_compute(grads, scale, batch_variance, epsilon, format_x)]


class BNInferGradKernelSpec:
    golden = staticmethod(bn_infer_grad_golden)
    third_party = {"torch": _TorchBNInferGrad}
    tolerance = _L0
