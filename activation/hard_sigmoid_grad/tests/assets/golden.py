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

主参考为 float64 开区间显式公式（与 kernel host 侧 double 边界推导及 ulp 外扩
同口径，见 ops-nn PR #13155 定版），ATen hardsigmoid_backward 降为 third_party
交叉对照：其仿射链在 fp32 下于 x≈±(1-β)/α 边界存在翻转，仅用于三方比对。

路由注册覆盖 kernel/aclnn/torch 全部 api_name（含 .default 变体）；
torch_graph 图入口供图模式使用，kernel 双态经 ttk.utilities 转换。
"""

import numpy as np

__spec__ = {
    "hard_sigmoid_grad": "HardSigmoidGradKernelSpec",
    "HardSigmoidGrad": "HardSigmoidGradKernelSpec",
    "aclnnHardsigmoidBackward": "HardSigmoidGradAclnnSpec",
    "torch.ops.aten.hardsigmoid_backward": "HardSigmoidGradTorchSpec",
    "torch.ops.aten.hardsigmoid_backward.default": "HardSigmoidGradTorchSpec",
}

_DEFAULT_ALPHA = 1.0 / 6.0
_DEFAULT_BETA = 0.5

_TOLERANCE = {
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "float32": {"standard": "mix_tolerance"},
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


def _aten_backward(grad, x, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA):
    """Third-party reference: ATen hardsigmoid_backward with affine transform.

    ATen is fixed to y/6+1/2; non-default alpha/beta are mapped through the
    affine transform. The mask chain runs in fp32, so boundary elements may
    disagree with the float64 main reference by one ulp — that divergence is
    the expected third-party signal under the comparison tolerance.
    """
    import torch

    alpha = float(alpha)
    beta = float(beta)
    transformed_x = torch.add(torch.mul(x, 6.0 * alpha), 6.0 * (beta - 0.5))
    transformed_grad = torch.mul(grad, 6.0 * alpha)
    return torch.ops.aten.hardsigmoid_backward(transformed_grad, transformed_x)


def _to_np(t):
    """Normalize numpy / torch (CPU or NPU) input to a numpy ndarray."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    import torch

    if isinstance(t, torch.Tensor):
        # torch.Tensor.numpy() rejects bfloat16; widen on the device-agnostic path.
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


def hard_sigmoid_grad_golden(
    grads, input_x, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA, **kwargs
):
    del kwargs
    import torch
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
    import torch

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


def torch_hard_sigmoid_grad_golden(grad_output, self, **kwargs):
    del kwargs
    import torch

    out_dtype = self.dtype if isinstance(self, torch.Tensor) else torch.float32
    g = (
        grad_output
        if isinstance(grad_output, torch.Tensor)
        else torch.as_tensor(grad_output)
    )
    x = self if isinstance(self, torch.Tensor) else torch.as_tensor(self)
    y = _masked(
        g.detach().to(torch.float32).cpu().numpy(),
        x.detach().to(torch.float32).cpu().numpy(),
        _DEFAULT_ALPHA,
        _DEFAULT_BETA,
    )
    return [torch.from_numpy(y).to(out_dtype)]


class _TorchHardSigmoidGrad:
    def __init__(self, *, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA):
        self.alpha = alpha
        self.beta = beta

    def __call__(instance, **kwargs):
        grad = kwargs.get("grads", kwargs.get("gradOutput", kwargs.get("grad_output")))
        x = kwargs.get("input_x", kwargs.get("self", kwargs.get("input")))
        return [_aten_backward(grad, x, instance.alpha, instance.beta)]


class _HardSigmoidGradGraph:
    def __init__(self, alpha=_DEFAULT_ALPHA, beta=_DEFAULT_BETA, **kwargs):
        del kwargs
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(self, grad_output, source):
        import torch

        out_dtype = source.dtype if isinstance(source, torch.Tensor) else torch.float32
        y = _masked(_to_np(grad_output), _to_np(source), self.alpha, self.beta)
        return torch.from_numpy(y).to(out_dtype)


class HardSigmoidGradKernelSpec:
    golden = staticmethod(hard_sigmoid_grad_golden)
    third_party = {"torch": _TorchHardSigmoidGrad}
    tolerance = _TOLERANCE


class HardSigmoidGradAclnnSpec:
    golden = staticmethod(aclnn_hard_sigmoid_grad_golden)
    third_party = {"torch": _TorchHardSigmoidGrad}
    tolerance = _TOLERANCE


class HardSigmoidGradTorchSpec:
    golden = staticmethod(torch_hard_sigmoid_grad_golden)
    third_party = {"torch": "torch.ops.aten.hardsigmoid_backward.default"}
    torch_graph = _HardSigmoidGradGraph
    tolerance = _TOLERANCE
