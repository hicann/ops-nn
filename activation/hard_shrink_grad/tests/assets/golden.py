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
"""TestSpec definitions for HardShrinkGrad on GEIR/Kernel, ACLNN, and Torch paths."""

import numpy

__spec__ = {
    "hard_shrink_grad": "HardShrinkGradKernelTestSpec",
    "HardShrinkGrad": "HardShrinkGradKernelTestSpec",
    "aclnnHardshrinkBackward": "AclnnHardshrinkBackwardTestSpec",
    "torch.ops.aten.hardshrink_backward": "TorchHardshrinkBackwardTestSpec",
}

_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
}


def _np_to_torch(arr):
    """numpy array to torch tensor, bit-exact for bfloat16 storage.

    bfloat16 has no native numpy dtype on the container side, so the 16-bit
    pattern is reinterpreted through an int16 view (same itemsize).
    """
    import torch

    if "bfloat16" in arr.dtype.name:
        return torch.from_numpy(arr.view(numpy.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(t, like):
    """torch tensor to numpy array, bit-exact for bfloat16 storage.

    ``like`` supplies the original container dtype (including the
    bfloat16 extension dtype) for the 16-bit reinterpretation.
    """
    import torch

    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(like.dtype)
    return t.numpy()


# Elements per CPU aten call: one below the vectorized-branch threshold of
# every supported dtype (torch 2.14 CPU: float32 16, float16 32, bfloat16 32),
# so each call takes the scalar branch.
_CHUNK = 15


def _hardshrink_grad_torch(grad, self, lambd):
    """CPU reference: direct call to the aten operator, chunked on CPU.

    ``torch.ops.aten.hardshrink_backward`` is the reference oracle (see
    ``__spec__``).  ``lambd`` is normalized to a Python number: the
    ACLNN binding hands the attribute over as a 0-dim tensor, and the
    aten schema wants a Scalar (f32 -> f64 -> f32 is an exact round
    trip, so the normalization cannot shift the threshold).

    On CPU the aten kernel dispatches to a vectorized branch above a
    per-dtype element threshold which zeroes NaN ``self`` entries
    instead of passing them through.  The CPU reference is evaluated in
    chunks below the vectorization threshold so element semantics are
    size-independent; NaN-free inputs keep the single direct call (the
    scalar and vectorized branches are bitwise identical there).
    """
    import torch

    lambd = float(lambd)
    aten = torch.ops.aten.hardshrink_backward
    if not grad.is_cpu:
        return aten(grad, self, lambd)
    shape = torch.broadcast_shapes(grad.shape, self.shape)
    numel = 1
    for dim in shape:
        numel *= dim
    if numel <= _CHUNK or (lambd == lambd and not bool(torch.isnan(self).any())):
        return aten(grad, self, lambd)
    grad_flat = grad.expand(shape).reshape(-1)
    self_flat = self.expand(shape).reshape(-1)
    chunks = [
        aten(grad_flat[i : i + _CHUNK], self_flat[i : i + _CHUNK], lambd)
        for i in range(0, numel, _CHUNK)
    ]
    return torch.cat(chunks).reshape(shape)


def hard_shrink_grad_golden(grad_out, self, lambd=0.5, **kwargs):
    """Kernel/GEIR golden: numpy arrays in, numpy array out.

    ``lambd`` defaults to 0.5 because testcases may omit the attribute
    (operator default 0.5, same as PyTorch).  numpy arrays only bridge
    the container boundary; the compute is the direct aten call on CPU.
    Inputs with broadcastable shapes combine per numpy broadcast rules.
    """
    grad_t = _np_to_torch(grad_out)
    self_t = _np_to_torch(self)
    result = _hardshrink_grad_torch(grad_t, self_t, lambd)
    return [_torch_to_np(result, grad_out)]


def aclnn_hardshrink_backward_golden(
    gradOutput, self, lambd=0.5, gradInput=None, **kwargs
):
    """Parameters follow ``aclnnHardshrinkBackwardGetWorkspaceSize``; CPU reference via direct aten call."""

    result = _hardshrink_grad_torch(gradOutput, self, lambd)
    if gradInput is not None:
        result = result.reshape(gradInput.shape).to(gradInput.dtype)
    return [result]


class HardshrinkBackwardCompose:
    """Third-party reference: direct call to the torch operator on the execution device."""

    def __init__(self, lambd=0.5, **kwargs):
        self.lambd = float(lambd)

    def __call__(self, *args, **kwargs):
        grad = kwargs.get("gradOutput", kwargs.get("gradients", kwargs.get("grad_out")))
        x = kwargs.get("self", kwargs.get("features"))
        if x is None and args:
            x = args[0]
        import torch

        return [torch.ops.aten.hardshrink_backward(grad, x, self.lambd)]


_THIRD_PARTY = {"torch": HardshrinkBackwardCompose}


class HardShrinkGradKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name``."""

    golden = staticmethod(hard_shrink_grad_golden)
    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnHardshrinkBackwardTestSpec:
    """ACLNN spec registered by CSV ``api_name``."""

    golden = staticmethod(aclnn_hardshrink_backward_golden)
    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchHardshrinkBackwardTestSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    def golden(grad_out, self, lambd=0.5, **kwargs):
        return [_hardshrink_grad_torch(grad_out, self, lambd)]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE
