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
"""Isolated public-PyTorch/autograd reference; default keeps TTK-promoted output."""

import torch
import torch.nn.functional as F

_RETURN_FINAL_DTYPE = False
_L2 = {
    name: {"standard": "cross_check", "level": "L2"}
    for name in ("float16", "float32", "bfloat16")
}


def _tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if "bfloat16" in x.dtype.name:
        import numpy

        return torch.from_numpy(x.view(numpy.int16)).view(torch.bfloat16)
    return torch.from_numpy(x)


def _numpy(x):
    x = x.detach().cpu()
    if x.dtype == torch.bfloat16:
        import ml_dtypes

        return x.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return x.numpy()


def _dtype(name):
    return getattr(torch, str(name).removeprefix("torch."))


def _first(mapping, names, default=None):
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return default


__spec__ = {
    "soft_margin_loss_grad": "SoftMarginLossGradKernelSpec",
    "aclnnSoftMarginLossBackward": "SoftMarginLossGradAclnnSpec",
}


def _reduction(x):
    if hasattr(x, "item"):
        x = x.item()
    return {"none": 0, "mean": 1, "sum": 2}[x] if isinstance(x, str) else int(x)


def _backward(go, source, target, reduction, aclnn=False):
    common = torch.promote_types(
        go.dtype, torch.promote_types(source.dtype, target.dtype)
    )
    go, source, target = (t.to(common) for t in (go, source, target))
    shape = torch.broadcast_shapes(go.shape, source.shape, target.shape)
    normalizer = source.numel()
    if aclnn:
        go, source, target = (
            t.expand(shape).contiguous() for t in (go, source, target)
        )
        normalizer = source.numel()
    with torch.inference_mode(False), torch.enable_grad():
        x = _normal(source.detach().expand(shape).contiguous()).requires_grad_(True)
        y = _normal(target.detach().expand(shape).contiguous())
        g = _normal(go.detach().expand(shape).contiguous())
        loss = F.soft_margin_loss(x, y, reduction="none")
        # Keep per-element upstream gradients and the route's original N.
        if _reduction(reduction) == 1 and normalizer:
            loss = loss / normalizer
        (result,) = torch.autograd.grad(loss, x, grad_outputs=g, create_graph=False)
    return result.detach()


class _TorchSoftMargin:
    aclnn_semantics = False

    def __init__(self, reduction="mean", **kwargs):
        self.reduction = reduction

    def __call__(adapter, *args, **kwargs):
        if adapter.aclnn_semantics:
            go = _first(kwargs, ("gradOutput",), args[0] if args else None)
            x = _first(kwargs, ("self",), args[1] if len(args) > 1 else None)
            y = _first(kwargs, ("target",), args[2] if len(args) > 2 else None)
        else:
            x = _first(kwargs, ("predict",), args[0] if args else None)
            y = _first(kwargs, ("label",), args[1] if len(args) > 1 else None)
            go = _first(kwargs, ("dout",), args[2] if len(args) > 2 else None)
        if any(t is None for t in (go, x, y)):
            raise ValueError("missing soft-margin reference inputs")
        return [
            _backward(
                go,
                x,
                y,
                kwargs.get("reduction", adapter.reduction),
                adapter.aclnn_semantics,
            )
        ]


class _TorchSoftMarginAclnn(_TorchSoftMargin):
    aclnn_semantics = True


class SoftMarginLossGradKernelSpec:
    @staticmethod
    def golden(predict, label, dout, reduction="mean", **kwargs):
        x, y, g = (_tensor(t) for t in (predict, label, dout))
        out = _backward(g, x, y, reduction)
        if _RETURN_FINAL_DTYPE:
            dtypes = kwargs.get("output_dtypes")
            out = out.to(_dtype(dtypes[0]) if dtypes else x.dtype)
        return [_numpy(out)]

    third_party = {"torch": _TorchSoftMargin}
    tolerance = _L2


class SoftMarginLossGradAclnnSpec:
    @staticmethod
    def golden(gradOutput, self, target, reduction=1, out=None, **kwargs):
        result = _backward(gradOutput, self, target, reduction, aclnn=True)
        if out is not None:
            result = result.reshape(out.shape)
            if _RETURN_FINAL_DTYPE:
                result = result.to(out.dtype)
        return [result]

    third_party = {"torch": _TorchSoftMarginAclnn}
    tolerance = _L2


def _normal(value):
    return value.clone() if torch.is_inference(value) else value


class _TorchSoftMarginTorch:
    """Retain the existing Torch dispatch entry with a public autograd reference."""

    def __init__(self, reduction=1, **kwargs):
        self.reduction = reduction

    def __call__(adapter, grad_output, *args, **kwargs):
        source = _first(kwargs, ("self", "source"), args[0] if args else None)
        target = _first(kwargs, ("target",), args[1] if len(args) > 1 else None)
        reduction = kwargs.get(
            "reduction", args[2] if len(args) > 2 else adapter.reduction
        )
        if source is None or target is None:
            raise ValueError("missing soft-margin Torch reference inputs")
        return [_backward(grad_output, source, target, reduction)]


class _SoftMarginLossGradGraph(torch.nn.Module):
    def forward(self, grad_output, source, target, reduction=1, **kwargs):
        return _backward(grad_output, source, target, reduction)


class SoftMarginLossGradTorchSpec:
    @staticmethod
    def golden(grad_output, self, target, reduction=1, out=None, **kwargs):
        result = _backward(grad_output, self, target, reduction)
        if out is not None:
            result = result.reshape(out.shape)
        return [result]

    third_party = {"torch": _TorchSoftMarginTorch}
    torch_graph = _SoftMarginLossGradGraph
    tolerance = _L2


# Legacy dispatch identifier only; execution uses the public forward and autograd.
__spec__["torch.ops.aten.soft_margin_loss_backward"] = "SoftMarginLossGradTorchSpec"
