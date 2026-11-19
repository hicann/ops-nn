#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""SmoothL1LossGrad 独立 Golden：raw/GEIR Kernel。

两方 Golden 和三方标杆都直接调用
``torch.ops.aten.smooth_l1_loss_backward`` 竞品算子
（reduction=none，beta=sigma，等价公式
gradient = clamp((predict - label) / sigma, -1, 1) * dout）。
"""

import math

import numpy as np

__spec__ = {
    "smooth_l1_loss_grad": "SmoothL1LossGradKernelSpec",
    "SmoothL1LossGrad": "SmoothL1LossGradKernelSpec",
}

_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}


def _validate_sigma(sigma):
    value = float(sigma)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"sigma must be finite and greater than zero, got {sigma}")
    return value


def _np_to_torch(array):
    """numpy ndarray -> torch tensor；bf16 无 numpy 原生 dtype，经 int16 位视图桥接。"""
    import torch

    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.name == "bfloat16":
        return (
            torch.from_numpy(contiguous.view(np.int16))
            .view(torch.bfloat16)
            .reshape(array.shape)
        )
    return torch.from_numpy(contiguous).reshape(array.shape)


def _torch_to_np(tensor, like):
    """torch tensor -> numpy ndarray；bf16 经 int16 位视图回投（dtype 对象复用输入侧，保持位级一致）。"""
    import torch

    if tensor.dtype == torch.bfloat16:
        return (
            tensor.contiguous()
            .view(torch.int16)
            .numpy()
            .view(like.dtype)
            .reshape(like.shape)
        )
    return tensor.numpy().reshape(like.shape)


def _compute(predict, label, dout, sigma):
    """Call the direct PyTorch Smooth L1 backward competitor."""
    import torch

    return torch.ops.aten.smooth_l1_loss_backward.default(
        dout,
        predict,
        label,
        0,  # Reduction::None
        sigma,
    )


def smooth_l1_loss_grad_golden(predict, label, dout, sigma=1.0, **kwargs):
    """CPU Golden; FP16/BF16 inputs are promoted before all computation."""
    del kwargs
    import torch

    is_torch = isinstance(predict, torch.Tensor)
    if is_torch:
        predict_tensor = predict
        label_tensor = label
        dout_tensor = dout
    else:
        predict_tensor = _np_to_torch(predict)
        label_tensor = _np_to_torch(label)
        dout_tensor = _np_to_torch(dout)
    output_dtype = predict_tensor.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        predict_tensor = predict_tensor.to(torch.float32)
        label_tensor = label_tensor.to(torch.float32)
        dout_tensor = dout_tensor.to(torch.float32)

    output = _compute(
        predict_tensor,
        label_tensor,
        dout_tensor,
        _validate_sigma(sigma),
    ).to(output_dtype)
    return [output if is_torch else _torch_to_np(output.detach().cpu(), predict)]


class _TorchSmoothL1LossGrad:
    """Direct Torch competitor without dtype or device conversion."""

    def __init__(self, sigma=1.0, **kwargs):
        del kwargs
        self.sigma = _validate_sigma(sigma)

    def __call__(self, predict, label, dout, **kwargs):
        del kwargs
        return [_compute(predict, label, dout, self.sigma)]


class SmoothL1LossGradKernelSpec:
    golden = staticmethod(smooth_l1_loss_grad_golden)
    third_party = {"torch": _TorchSmoothL1LossGrad}
    tolerance = _L2
