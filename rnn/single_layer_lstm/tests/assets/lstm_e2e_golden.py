# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK LSTM forward: CPU FP64 golden and same-dtype CUDA reference."""

import contextlib
import warnings

import torch

__spec__ = {"npu_vf_testkits.lstm": "VfLstmTestSpec"}


def _wide(value):
    return value.detach().to(device="cpu", dtype=torch.float64)


def _fp64(args):
    x, hx, params, *attributes = args
    return (_wide(x), [_wide(t) for t in hx], [_wide(t) for t in params], *attributes)


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


def forward_golden(*args, **kwargs):
    return _forward(_fp64(args))


class _ForwardCuda:
    def __call__(
        self,
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        **kwargs,
    ):
        if input.device.type != "cuda":
            raise ValueError("The third-party reference must run on CUDA")
        args = (
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
        with _backend(args):
            return _forward(args)


class VfLstmTestSpec:
    golden = staticmethod(forward_golden)
    third_party = {"torch": _ForwardCuda}
    tolerance = {
        name: {"standard": "cross_check", "level": "L2"}
        for name in ("float32", "float16", "bfloat16")
    }
