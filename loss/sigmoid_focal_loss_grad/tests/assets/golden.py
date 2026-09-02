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

"""SigmoidFocalLossGrad Golden for Kernel and GEIR paths.

The calculation below follows, statement by statement, the Golden at:
LongTailInfo/8August requirment/sigmoid_focal_loss_grad/golden/
sigmoid_focal_loss_grad_golden.py

Source SHA256: c10e2a95a87e50702f73091e3f6c57d9919496975c839e6fb978907c35faf35b.
The TestSpec structure follows LongTailInfo/Sinh/golden文件/sinh.py. Only the
raw ``sigmoid_focal_loss_grad`` name is registered because this operator is
tested through Kernel direct-call and GEIR paths only. GEIR reuses the Kernel
spec. The Torch small-operator composition is also registered as the
third-party reference, with the L0 cross-check standard.
"""

import torch


__spec__ = {
    "sigmoid_focal_loss_grad": "SigmoidFocalLossGradKernelSpec",
}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}

# TBE Constant.CONST_FP_MIN (the minimum normal float32 value).
CONST_FP_MIN = 1.17549435e-38


def _to_torch(x, name, float_only=True):
    """Convert a Kernel/GEIR numpy input to torch while preserving dtype."""
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        from ttk.utilities.dtypes import numpy_to_torch_tensor

        tensor = numpy_to_torch_tensor(x)
    if float_only and not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be FP16/FP32/BF16, got {tensor.dtype}")
    return tensor


def _orig_dtype_str(tensor):
    """Return the source Golden output carrier dtype."""
    if tensor.dtype == torch.float16:
        return "float16"
    return "float32"


def _compute(
    pred,
    target,
    dout,
    weight=None,
    alpha=0.25,
    gamma=2.0,
    reduction="mean",
):
    """Compute SigmoidFocalLossGrad using the frozen TBE operation order."""
    pred_t = _to_torch(pred, "pred", float_only=True)
    target_t = _to_torch(target, "target", float_only=False)
    dout_t = _to_torch(dout, "dout", float_only=True)
    has_weight = weight is not None
    weight_t = _to_torch(weight, "weight", float_only=True) if has_weight else None

    # Shape and reduction checks follow the source Golden.
    if pred_t.dim() != 2:
        raise ValueError(f"pred must be 2D [N, C], got rank {pred_t.dim()}")
    if target_t.dim() != 2 or tuple(pred_t.shape) != tuple(target_t.shape):
        raise ValueError(f"target must be 2D with shape {tuple(pred_t.shape)}")
    if tuple(pred_t.shape) != tuple(dout_t.shape) and dout_t.dim() != 0:
        raise ValueError(
            f"dout shape {tuple(dout_t.shape)} must equal pred shape "
            f"{tuple(pred_t.shape)} or be scalar"
        )
    if has_weight and tuple(weight_t.shape) != tuple(pred_t.shape):
        raise ValueError(f"weight shape must equal pred shape {tuple(pred_t.shape)}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be mean/sum/none, got {reduction}")

    orig_dtype_str = _orig_dtype_str(pred_t)

    # The operation sequence below mirrors sigmoid_focal_loss_grad_compute.
    pred_f = pred_t.to(torch.float32)
    target_f = target_t.to(torch.float32)
    dout_f = dout_t.to(torch.float32)
    if has_weight:
        weight_f = weight_t.to(torch.float32)

    probs = torch.sigmoid(pred_f)
    probs_nadd = torch.sub(1.0, probs)

    probs_clamp = torch.clamp(probs, min=CONST_FP_MIN)
    probs_nadd_clamp = torch.clamp(probs_nadd, min=CONST_FP_MIN)

    probs_log = torch.log(probs_clamp)
    probs_nlog = torch.log(probs_nadd_clamp)

    pow_y = torch.exp((gamma + 1.0) * probs_nlog)
    dpos_front = torch.mul(-alpha, pow_y)

    pow_v = torch.exp(gamma * probs_nlog)
    dpos_back = torch.mul(pow_v, probs_log)
    dpos_back = torch.mul(dpos_back, probs)
    dpos_back = torch.mul(dpos_back, gamma * alpha)

    dpos = torch.add(dpos_front, dpos_back)

    pow_n = torch.exp(gamma * probs_log)
    dneg_front = torch.mul(pow_n, probs_nadd)
    dneg_front = torch.mul(dneg_front, probs_nlog)
    dneg_front = torch.mul(dneg_front, gamma * (alpha - 1.0))

    pow_q = torch.exp((gamma + 1.0) * probs_log)
    dneg_back = torch.mul(1.0 - alpha, pow_q)

    dneg = torch.add(dneg_front, dneg_back)

    target_nadd = torch.sub(1.0, target_f)
    grad_front = torch.mul(dpos, target_nadd)
    grad_back = torch.mul(dneg, target_f)
    result = torch.add(grad_front, grad_back)

    if has_weight:
        result = torch.mul(result, weight_f)

    result = torch.mul(result, dout_f)

    if reduction == "mean":
        element_count = 1.0
        for dim in pred_f.shape:
            element_count *= dim
        coefficient = 1.0 / element_count if element_count != 0.0 else 0.0
        result = torch.mul(result, coefficient)

    output_dtype = torch.float16 if orig_dtype_str == "float16" else torch.float32
    output = result.to(output_dtype)

    return output


def sigmoid_focal_loss_grad_golden(
    pred,
    target,
    dout,
    weight=None,
    alpha=0.25,
    gamma=2.0,
    reduction="mean",
    **kwargs,
):
    """Return the Kernel/GEIR Golden in the same carrier type as ``pred``."""
    del kwargs
    is_torch = isinstance(pred, torch.Tensor)
    output = _compute(pred, target, dout, weight, alpha, gamma, reduction)
    if is_torch:
        return [output]

    from ttk.utilities.dtypes import torch_to_numpy_tensor

    return [torch_to_numpy_tensor(output.detach().cpu())]


class _TorchSigmoidFocalLossGrad:
    """Torch small-operator third-party reference for the raw selector semantics."""

    def __init__(self, *, alpha=0.25, gamma=2.0, reduction="mean", **kwargs):
        del kwargs
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, pred, target, dout, weight=None, **kwargs):
        del kwargs
        return [
            _compute(
                pred,
                target,
                dout,
                weight,
                self.alpha,
                self.gamma,
                self.reduction,
            )
        ]


class SigmoidFocalLossGradKernelSpec:
    golden = staticmethod(sigmoid_focal_loss_grad_golden)
    third_party = {"torch": _TorchSigmoidFocalLossGrad}
    tolerance = _L0
