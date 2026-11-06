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

"""SoftShrinkGrad 独立 Golden：Kernel、ACLNN、Torch。

The operator computes the softshrink backward gradient:
``y = gradOutput`` when ``|self| > lambd`` else ``0``, with NaN positions
passing ``gradOutput`` through (aligned with PyTorch ATen semantics).

Reference oracle: the user-layer ``torch.nn.functional.softshrink`` autograd
backward. Customers call the torch functional API, not the internal aten op;
its registered derivative is exactly this gradient, so the reference is
computed through the same user-layer path the delivery exposes.
"""

import torch
import torch.nn.functional as F

__spec__ = {
    "soft_shrink_grad": "SoftShrinkGradKernelSpec",
    "aclnnSoftshrinkBackward": "AclnnSoftshrinkBackwardSpec",
    "torch.ops.aten.softshrink_backward": "TorchSoftshrinkBackwardSpec",
    "torch.nn.functional.softshrink": "TorchSoftshrinkSpec",
}


_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
}


def _aten_softshrink_backward(grad_output, x, lambd=0.5):
    """Call the ATen competitor after adapting TTK's scalar tensor."""

    return torch.ops.aten.softshrink_backward(grad_output, x, float(lambd))


def aclnn_softshrink_backward_golden(
    gradOutput, self, lambd=0.5, gradInput=None, **kwargs
):
    """Parameters follow ``aclnnSoftshrinkBackwardGetWorkspaceSize``."""

    del kwargs
    result = _aten_softshrink_backward(gradOutput, self, lambd)
    if gradInput is not None:
        result = result.reshape(gradInput.shape).to(gradInput.dtype)
    return [result]


class SoftShrinkGradGraph(torch.nn.Module):
    """Graph module for the ATen softshrink backward path."""

    def __init__(self, *args, **kwargs):
        # 兜底：框架侧若重建过 OpOverloadPacket（如 resolve_function 曾直呼
        # __getattr__ 顶掉缓存），模块级注册的 converter 会随旧对象丢失，
        # 此处在编译前按 _softshrink_grad_patched 标记幂等重注册。
        super().__init__(*args, **kwargs)
        _register_graph_runtime_support()

    def forward(self, grad_output, input_self=None, lambd=0.5):
        return _aten_softshrink_backward(grad_output, input_self, lambd)


class SoftShrinkGradKernelThirdParty:
    """third_party (Kernel/GEIR): parameter names match soft_shrink_grad def.cpp
    (input_grad / input_x / lambd)."""

    def __call__(self, input_grad, input_x, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(input_grad, input_x, lambd)]


class SoftShrinkGradAclnnThirdParty:
    """third_party (ACLNN): parameter names match aclnnSoftshrinkBackward's
    header (gradOutput / self / lambd). ``self`` is declared as an explicit
    parameter: on a bound ``__call__`` the framework binds the pool entry named
    ``self`` by parameter name (aten-style reserved-name handling)."""

    def __call__(instance, gradOutput, self=None, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(gradOutput, self, lambd)]


class SoftShrinkGradTorchThirdParty:
    """third_party (E2E): parameter names match the aten schema
    (grad_output / self / lambd); ``self`` is declared as an explicit
    parameter, bound by name like the ACLNN adapter."""

    def __call__(instance, grad_output, self=None, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(grad_output, self, lambd)]


class SoftShrinkGradKernelSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name``."""

    golden = "torch.ops.aten.softshrink_backward"
    third_party = {"torch": SoftShrinkGradKernelThirdParty}
    tolerance = _L2


class AclnnSoftshrinkBackwardSpec:
    """ACLNN spec registered by CSV ``api_name``."""

    golden = staticmethod(aclnn_softshrink_backward_golden)
    third_party = {"torch": SoftShrinkGradAclnnThirdParty}
    tolerance = _L2


class TorchSoftshrinkBackwardSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    def golden(grad_out, x, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(grad_out, x, lambd)]

    third_party = {"torch": SoftShrinkGradTorchThirdParty}
    torch_graph = SoftShrinkGradGraph
    tolerance = _L2


def _torch_softshrink_forward_backward(input, lambd=0.5):
    """Torch public-API path: forward via ``F.softshrink`` and backward via the
    autograd engine (``torch.func.vjp`` with an ones cotangent — same seed as
    ttk's ``eager_backward`` protocol).

    Returns ``(forward_output, grad_input)``; works eager (CPU golden / GPU
    third_party) and under ``torch.compile`` (the vjp traces into a joint GE
    graph that lowers ``aten::softshrink`` + ``aten::softshrink_backward``)."""

    y, vjp_fn = torch.func.vjp(lambda t: F.softshrink(t, lambd), input)
    grad = vjp_fn(torch.ones_like(y))[0]
    return y, grad


def torch_softshrink_golden(input, lambd=0.5, **kwargs):
    """Torch-interface golden (CPU): [forward, grad] — same output protocol as
    ttk's backward_mode (grads appended at the tail)."""

    del kwargs
    y, grad = _torch_softshrink_forward_backward(input, lambd)
    return [y, grad]


class SoftShrinkVjpGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _register_graph_runtime_support()

    def forward(self, input, lambd=0.5):
        return _torch_softshrink_forward_backward(input, lambd)


class TorchSoftshrinkThirdParty:
    """third_party (XPU/E2E): parameter names match
    ``torch.nn.functional.softshrink`` (input / lambd); executes the
    public-API forward+backward on the target device (GPU server)."""

    def __call__(instance, input=None, lambd=0.5, **kwargs):
        del kwargs
        y, grad = _torch_softshrink_forward_backward(input, lambd)
        return [y, grad]


class TorchSoftshrinkSpec:
    """Torch public-interface spec registered by CSV ``api_name``
    (``torch.nn.functional.softshrink.backward`` — ttk strips the suffix and
    enables backward_mode; equal-shape rows only, see module docstring)."""

    golden = staticmethod(torch_softshrink_golden)
    third_party = {"torch": TorchSoftshrinkThirdParty}
    torch_graph = SoftShrinkVjpGraph
    tolerance = _L2


def _register_graph_runtime_support():
    try:
        try:
            import torchair._ge_concrete_graph.ge_converter.aten.softshrink  # noqa: F401
        except Exception:
            pass
        import torchair._ge_concrete_graph.ge_converter.aten.softshrink_backward  # noqa: F401
        from torchair._ge_concrete_graph.fx2ge_converter import (
            register_fx_node_ge_converter,
        )
        from torchair._ge_concrete_graph import ge_apis as _ge
        from torchair._ge_concrete_graph.utils import (
            dtype_promote,
            torch_type_to_ge_type,
        )
        from torchair.ge._ge_graph import Tensor as _GeTensor
    except Exception:
        return

    if getattr(
        torch.ops.aten.softshrink_backward.default, "_softshrink_grad_patched", False
    ):
        return

    @register_fx_node_ge_converter(torch.ops.aten.softshrink.default)
    def _conveter_aten_softshrink_default(self, lambd, meta_outputs=None):
        """NB: aten::softshrink(Tensor self, Scalar lambd) -> Tensor"""
        if isinstance(lambd, _GeTensor):
            raise NotImplementedError(
                "torch.ops.aten.softshrink.default ge_converter is "
                "not implemented when lambd is tensor!"
            )
        if self.dtype in (
            torch_type_to_ge_type(torch.float16),
            torch_type_to_ge_type(torch.bfloat16),
        ):
            y32 = _ge.SoftShrink(
                dtype_promote(self, target_dtype=torch.float32), lambd=float(lambd)
            )
            return _ge.CastLike(y32, self)
        return _ge.SoftShrink(self, lambd=float(lambd))

    @register_fx_node_ge_converter(torch.ops.aten.softshrink_backward.default)
    def _conveter_aten_softshrink_backward_default(
        grad_output, self, lambd, meta_outputs=None
    ):
        """NB: aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor"""
        if isinstance(lambd, _GeTensor):
            raise NotImplementedError(
                "torch.ops.aten.softshrink_backward.default ge_converter is "
                "not implemented when lambd is tensor!"
            )
        lhs, rhs = grad_output.symsize, self.symsize
        if str(lhs) != str(rhs):
            common = list(torch.broadcast_shapes(tuple(lhs), tuple(rhs)))
            if str(common) == str(lhs):
                self = _ge.BroadcastTo(self, _ge.Shape(grad_output))
            elif str(common) == str(rhs):
                grad_output = _ge.BroadcastTo(grad_output, _ge.Shape(self))
            elif all(not isinstance(d, torch.SymInt) for d in common):
                grad_output = _ge.BroadcastToD(
                    grad_output, shape=[int(d) for d in common]
                )
                self = _ge.BroadcastToD(self, shape=[int(d) for d in common])
            else:
                raise NotImplementedError(
                    "torch.ops.aten.softshrink_backward.default ge_converter "
                    "cannot align mixed symbolic shapes!"
                )
        return _ge.SoftShrinkGrad(grad_output, self, lambd=float(lambd))

    torch.ops.aten.softshrink_backward.default._softshrink_grad_patched = True


_register_graph_runtime_support()


def _register_softshrink_meta():
    try:

        @torch.library.impl("aten::softshrink", "Meta")
        def _softshrink_meta(self, lambd=0.5):
            return torch.empty_like(self)

    except Exception:
        pass

    try:

        @torch.library.impl("aten::softshrink_backward", "Meta")
        def _softshrink_backward_meta(grad_output, self, lambd=0.5):
            shape = torch.broadcast_shapes(tuple(grad_output.shape), tuple(self.shape))
            return torch.empty(
                shape, dtype=torch.result_type(grad_output, self), device="meta"
            )

    except Exception:
        pass


_register_softshrink_meta()
