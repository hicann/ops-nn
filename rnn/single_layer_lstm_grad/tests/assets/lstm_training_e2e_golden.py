# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK LSTM forward/backward with explicit dy, dh_n and dc_n inputs.

``lstm_training_e2e_golden.lstm_train`` is a Python entry point for a complete
LSTM forward/backward run, rather than a built-in PyTorch operator or a
standalone backward call. Its semantics are ``torch._VF.lstm(..., train=True)``
followed by ``torch.autograd.grad`` with the supplied upstream gradients.
The NPU call uses the installed torch_npu/op-plugin dispatch and autograd;
it requires versions that support the tested LSTM configuration. This module
does not override that dispatch or load a private operator extension.

Tensor arguments are input, hx=[h0, c0], params, dy, dh_n and dc_n, followed by
has_biases, num_layers, dropout, train, bidirectional and batch_first.
dy, dh_n and dc_n are independent test inputs corresponding to y, h_n and c_n;
the adapter does not generate them. Forward execution produces the saved states
consumed by backward. This test requires train=True and dropout=0.
The returned tuple is (y, h_n, c_n, dx, dh0, dc0, *dparams), with parameter
gradients in the same order as params.

The __spec__ key matches the adapter API name in the test-case CSV.
The harness supplies identical declared-dtype inputs to all three backends.
Only the CPU golden widens those supplied values to FP64.
TTK places the third-party inputs on the configured reference device.
"""

import contextlib
import warnings
from typing import List

import torch

__spec__ = {"lstm_training_e2e_golden.lstm_train": "VfLstmTrainTestSpec"}


def _wide(value):
    return value.detach().to(device="cpu", dtype=torch.float64)


def _leaf(value):
    return value.detach().clone().requires_grad_(True)


def _fp64(args):
    if len(args) != 12:
        raise ValueError(
            "Expected input, hx, params, dy, dh_n, dc_n and six VF attributes"
        )
    x, hx, params, dy, dh_n, dc_n, *attributes = args
    for name, value in zip(("dy", "dh_n", "dc_n"), (dy, dh_n, dc_n)):
        if value.dtype != x.dtype or value.device != x.device:
            raise ValueError(
                f"{name} must have the input dtype and device before golden promotion"
            )
    return (
        _wide(x),
        [_wide(t) for t in hx],
        [_wide(t) for t in params],
        _wide(dy),
        _wide(dh_n),
        _wide(dc_n),
        *attributes,
    )


def _backend(args):
    # CUDA ATen accepts H=0; cuDNN rejects it. Do not change device, dtype or TF32 policy.
    if args[0].device.type == "cuda" and args[1][0].shape[-1] == 0:
        warnings.warn(
            "H=0 reference uses CUDA ATen with cuDNN disabled",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.backends.cudnn.flags(
            enabled=False,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        )
    return contextlib.nullcontext()


def _forward(args):
    if len(args) != 9:
        raise ValueError("Expected the nine arguments of aten::lstm.input")
    if float(args[5]) != 0.0:
        raise ValueError(
            "Cross-backend dropout masks are not synchronized; dropout must be zero"
        )
    return tuple(torch._VF.lstm(*args))


def _train(args):
    if len(args) != 12:
        raise ValueError(
            "Expected input, hx, params, dy, dh_n, dc_n and six VF attributes"
        )
    x, hx, params, dy, dh_n, dc_n, *attributes = args
    if not attributes[3]:
        raise ValueError("lstm_train requires train=True")
    leaves = [_leaf(x), *(_leaf(t) for t in hx), *(_leaf(t) for t in params)]
    call = (leaves[0], leaves[1:3], leaves[3:], *attributes)
    with torch.enable_grad(), _backend(call):
        outputs = _forward(call)
        if not any(value.requires_grad for value in outputs):
            raise RuntimeError("Reference returned no autograd graph")
        upstream = (dy, dh_n, dc_n)
        for name, gradient, value in zip(("dy", "dh_n", "dc_n"), upstream, outputs):
            if (
                gradient.shape != value.shape
                or gradient.dtype != value.dtype
                or gradient.device != value.device
            ):
                raise ValueError(
                    f"{name} must match its output's shape, dtype and device"
                )
        gradients = torch.autograd.grad(outputs, leaves, grad_outputs=upstream)
    return tuple(value.detach() for value in outputs) + gradients


def backward_golden(*args, **kwargs):
    return _train(_fp64(args))


def lstm_train(
    input: torch.Tensor,
    hx: List[torch.Tensor],
    params: List[torch.Tensor],
    dy: torch.Tensor,
    dh_n: torch.Tensor,
    dc_n: torch.Tensor,
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
):
    """Run native VF forward and autograd with the supplied upstream gradients."""
    return _train(
        (
            input,
            hx,
            params,
            dy,
            dh_n,
            dc_n,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
    )


class TorchRefImpl:
    def __call__(
        self,
        input,
        hx,
        params,
        dy,
        dh_n,
        dc_n,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        **kwargs,
    ):
        args = (
            input,
            hx,
            params,
            dy,
            dh_n,
            dc_n,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
        return _train(args)


class VfLstmTrainTestSpec:
    golden = staticmethod(backward_golden)
    third_party = {"torch": TorchRefImpl}
    tolerance = {
        name: {"standard": "cross_check", "level": "L2"}
        for name in ("float32", "float16", "bfloat16")
    }
