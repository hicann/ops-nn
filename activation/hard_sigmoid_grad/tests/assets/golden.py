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

"""HardSigmoidGrad 独立 Golden。

kernel/aclnn 路由：float64 开区间显式公式主参考。
torch 路由：torch.nn.functional.hardsigmoid，backward golden 与 third_party
各为独立参考（float64 显式公式 vs 用户层 autograd）。浮点容差 cross_check L1。
"""

import numpy as np
import torch

__spec__ = {
    "hard_sigmoid_grad": "HardSigmoidGradKernelSpec",
    "HardSigmoidGrad": "HardSigmoidGradKernelSpec",
    "aclnnHardsigmoidBackward": "HardSigmoidGradAclnnSpec",
    "torch.nn.functional.hardsigmoid": "HardSigmoidGradTorchSpec",
}

_DEFAULT_ALPHA = 1.0 / 6.0
_DEFAULT_BETA = 0.5

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _masked(grads, input_x, alpha, beta):
    """grads * alpha where 0 < alpha*input_x + beta < 1 (open interval, float64)."""
    alpha_d = np.float64(alpha)
    beta_d = np.float64(beta)
    x = np.asarray(input_x).astype(np.float64)
    grad = np.asarray(grads).astype(np.float64)
    inner = alpha_d * x + beta_d
    keep = (inner > 0.0) & (inner < 1.0)
    return np.where(keep, grad * alpha_d, 0.0)


def _to_np(t):
    """Normalize numpy / torch (CPU or NPU) input to a numpy ndarray."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()  # bf16 needs widening
    return np.asarray(t)


def _to_out_dtype(input):
    """Derive the output dtype from the input tensor/array."""
    if isinstance(input, torch.Tensor):
        return input.dtype
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input).dtype
    return torch.float32


def hard_sigmoid_grad_golden(
    grads, input_x, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA, **kwargs
):
    del kwargs
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(grads, torch.Tensor)
    grad_t = grads if is_torch else numpy_to_torch_tensor(grads)
    x_t = (
        input_x if isinstance(input_x, torch.Tensor) else numpy_to_torch_tensor(input_x)
    )
    out_dtype = x_t.dtype
    y = _masked(
        grad_t.detach().to(torch.float32).cpu().numpy(),
        x_t.detach().to(torch.float32).cpu().numpy(),
        alpha,
        beta,
    )
    result = torch.from_numpy(y).to(out_dtype)
    return [result if is_torch else torch_to_numpy_tensor(result.detach().cpu())]


def aclnn_hard_sigmoid_grad_golden(gradOutput, self, out=None, **kwargs):
    del kwargs
    out_dtype = torch.float32
    if isinstance(out, torch.Tensor):
        out_dtype = out.dtype
    elif out is not None:
        out_dtype = torch.from_numpy(np.asarray(out)).dtype
    y = _masked(_to_np(gradOutput), _to_np(self), _DEFAULT_ALPHA, _DEFAULT_BETA)
    result = torch.from_numpy(y).to(out_dtype)
    if out is not None:
        result = result.reshape(out.shape)
    return [result]


def _userlayer_backward(grad, x, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA):
    """User-layer backward: F.hardsigmoid + autograd, native dtype (no upcast).

    Non-default alpha/beta map through x' = 6*alpha*x + 6*(beta-0.5); the
    open interval 0 < alpha*x + beta < 1 equals x' in (-3, 3), and the chain
    rule yields grad * alpha * mask — the operator semantics element-wise.
    """
    import torch.nn.functional as F

    alpha = float(alpha)
    beta = float(beta)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not x.is_floating_point():
        return torch.zeros_like(x)
    x_t = x.detach().clone().requires_grad_(True)
    if not torch.is_tensor(grad) and grad is not None:
        grad = torch.as_tensor(grad)
    transformed_x = torch.add(torch.mul(x_t, 6.0 * alpha), 6.0 * (beta - 0.5))
    out = F.hardsigmoid(transformed_x)
    if grad is None:
        seed = torch.ones_like(out)
    elif grad.shape != out.shape:
        raise ValueError(
            f"third_party grad shape mismatch: {tuple(grad.shape)} vs out {tuple(out.shape)}"
        )
    else:
        seed = grad.detach().to(out.dtype).to(out.device)
    (grad_x,) = torch.autograd.grad(out, x_t, grad_outputs=seed)
    return grad_x


def hardsigmoid_backward_golden(input, inplace=None, **kwargs):
    """Torch route backward golden: float64 explicit formula, [forward, grad]."""
    del kwargs
    out_dtype = _to_out_dtype(input)
    x64 = _to_np(input).astype(np.float64)

    is_0d = np.ndim(x64) == 0
    x64 = np.atleast_1d(x64)
    inner = x64 * _DEFAULT_ALPHA + _DEFAULT_BETA
    forward64 = np.clip(inner, 0.0, 1.0)
    mask = (inner > 0.0) & (inner < 1.0)
    grad64 = mask * _DEFAULT_ALPHA
    if is_0d:
        forward64 = forward64.reshape(())
        grad64 = grad64.reshape(())

    return [
        torch.from_numpy(forward64).to(out_dtype),
        torch.from_numpy(grad64).to(out_dtype),
    ]


def _resolve_grad_x(instance, gradOutput=None, self=None, *args, **kwargs):
    grad = kwargs.get("grads")
    if grad is None:
        grad = kwargs.get("grad_output")
    if grad is None:
        grad = gradOutput
    x = kwargs.get("input_x")
    if x is None:
        x = kwargs.get("input")
    if x is None:
        x = self
    return grad, x


class _TorchHardSigmoidGrad:
    # bind_params excludes "self" from kwargs; declare both binding name sets.
    def __init__(self, *, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA):
        self.alpha = alpha
        self.beta = beta

    def __call__(instance, gradOutput=None, self=None, *args, **kwargs):
        grad, x = _resolve_grad_x(instance, gradOutput, self, *args, **kwargs)
        return [_userlayer_backward(grad, x, instance.alpha, instance.beta)]


class _TorchHardSigmoidGradBackward:
    """Torch-route third_party: [forward, grad] with ones cotangent (repo convention)."""

    def __call__(instance, input=None, *args, **kwargs):
        del args, kwargs
        import torch.nn.functional as F

        if not torch.is_tensor(input):
            input = torch.as_tensor(input)
        if not input.is_floating_point():
            zeros = torch.zeros_like(input)
            return [zeros, zeros]
        out_dtype = input.dtype
        x_t = input.detach().clone().requires_grad_(True)
        forward = F.hardsigmoid(x_t)
        (grad_x,) = torch.autograd.grad(
            forward, x_t, grad_outputs=torch.ones_like(forward)
        )
        return [forward.detach().to(out_dtype), grad_x.detach().to(out_dtype)]


class HardSigmoidGradKernelSpec:
    golden = staticmethod(hard_sigmoid_grad_golden)
    third_party = {"torch": _TorchHardSigmoidGrad}
    tolerance = _TOLERANCE


class HardSigmoidGradAclnnSpec:
    golden = staticmethod(aclnn_hard_sigmoid_grad_golden)
    third_party = {"torch": _TorchHardSigmoidGrad}
    tolerance = _TOLERANCE


class HardSigmoidGradTorchSpec:
    golden = staticmethod(hardsigmoid_backward_golden)
    third_party = {"torch": _TorchHardSigmoidGradBackward}
    tolerance = _TOLERANCE
