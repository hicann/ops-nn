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

"""HardSigmoidGrad 的 TTK 多路径 TestSpec golden。

参考 README.md 与 op_graph/hard_sigmoid_grad_proto.h:
    y = grads * alpha            if 0 < alpha * input_x + beta < 1
    y = 0                        otherwise
    alpha 默认 1/6（float32 0.16666667），beta 默认 0.5。

掩码在 float64 中求值（与 kernel 的 host 侧 double 边界推导一致），
避免 fp32 乘加舍入在 x 略小于 (1-beta)/alpha（默认 3.0）时翻转判定。
"""

__spec__ = {
    "HardSigmoidGrad": "HardSigmoidGradKernelSpec",
    "hard_sigmoid_grad": "HardSigmoidGradKernelSpec",
    "aclnnHardsigmoidBackward": "AclnnHardsigmoidBackwardSpec",
    "torch.ops.aten.hardsigmoid_backward": "TorchHardSigmoidBackwardSpec",
}

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None

_DEFAULT_ALPHA = 0.16666667  # = 1/6 in float32
_DEFAULT_BETA = 0.5


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
    import torch

    if isinstance(t, torch.Tensor):
        # torch.Tensor.numpy() rejects bfloat16; widen on the device-agnostic path.
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


class HardSigmoidGradKernelSpec:
    """Kernel (REG_OP) path: numpy inputs, attributes alpha/beta via kwargs."""

    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }

    def golden(grads, input_x, **kwargs):
        alpha = kwargs.get("alpha", _DEFAULT_ALPHA)
        beta = kwargs.get("beta", _DEFAULT_BETA)
        y = _masked(_to_np(grads), _to_np(input_x), alpha, beta)
        out_dtype = np.asarray(input_x).dtype
        if _bf16 is not None and out_dtype == _bf16:
            return [y.astype(_bf16)]
        return [y.astype(out_dtype, copy=False)]


class AclnnHardsigmoidBackwardSpec:
    """ACLNN path: torch tensors (gradOutput, self, out); result in out dtype.

    The aclnn entry casts both inputs to PromoteType(gradOutput, self), runs the
    op with alpha = 1/6 and beta = 0.5 (fixed), then casts the result to out.
    """

    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }

    def golden(gradOutput, self, out=None, **kwargs):
        import torch

        # The aclnn runner binds only CSV inputs (+optional attrs); out may be
        # absent, a torch tensor, or a numpy array. Normalize its dtype through
        # the same widening rules used for the inputs.
        out_dtype = torch.float32
        if isinstance(out, torch.Tensor):
            out_dtype = out.dtype
        elif out is not None:
            out_dtype = torch.from_numpy(np.asarray(out)).dtype
        y = _masked(_to_np(gradOutput), _to_np(self), 1.0 / 6.0, 0.5)
        return [torch.from_numpy(y).to(out_dtype)]


class TorchHardSigmoidBackwardSpec:
    """Torch eager/graph e2e path: torch tensors (gradOutput, self).

    aten.hardsigmoid_backward has no alpha/beta arguments; its fixed
    |x| < 3 mask equals the default alpha = 1/6, beta = 0.5 semantics, so the
    golden delegates to the aten CPU reference directly.
    """

    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }

    def golden(gradOutput, self, **kwargs):
        import torch

        g = (
            gradOutput
            if isinstance(gradOutput, torch.Tensor)
            else torch.as_tensor(gradOutput)
        )
        x = self if isinstance(self, torch.Tensor) else torch.as_tensor(self)
        out_dtype = torch.result_type(g, x)
        y = torch.ops.aten.hardsigmoid_backward(
            g.to(torch.float32), x.to(torch.float32)
        )
        return [y.to(out_dtype)]

    class TorchThirdParty:
        """torch third-party reference: native aten.hardsigmoid_backward."""

        def __init__(self, **kwargs):
            pass

        def __call__(self, gradOutput, input_x, **kwargs):
            import torch

            return [torch.ops.aten.hardsigmoid_backward(gradOutput, input_x)]

    third_party = {"torch": TorchThirdParty}
