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
"""Golden reference for ClaGateBackward using the TTK TestSpec format.

Golden：
  前向 merge:  merged = sigmoid(z_g)⊙O_g + sigmoid(z_l)⊙O_l
  反向:        torch.autograd.grad(merged, (O_g, O_l, z_g, z_l), grad_merged)
等价于手写公式：
  d_O_g = G ⊙ s_g
  d_O_l = G ⊙ s_l
  d_z_g = (Σ_D G⊙O_g) ⊙ s_g ⊙ (1 - s_g)
  d_z_l = (Σ_D G⊙O_l) ⊙ s_l ⊙ (1 - s_l)
其中 Σ_D 只沿 head_dim 归约。归约在 FP32 下进行以匹配 kernel 的 FP32 累加。
"""

import numpy as np
import torch


__spec__ = {
    "cla_gate_backward": "cla_gate_backward_golden",
}

__golden__ = {
    "kernel": {"cla_gate_backward": "cla_gate_backward_golden"},
}

_TOLERANCE = {
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}


def _numpy_dtype(dtype):
    """Resolve TTK dtype values, including NumPy's optional bfloat16 dtype."""
    name = getattr(dtype, "name", str(dtype)).lower()
    if name in ("bf16", "bfloat16"):
        try:
            from ml_dtypes import bfloat16
        except ImportError as exc:
            raise RuntimeError(
                "ClaGateBackward bfloat16 golden requires the optional ml-dtypes package"
            ) from exc
        return bfloat16
    return np.dtype(dtype)


def _to_torch(array):
    """Convert a contiguous NumPy tensor to torch, promoting fp16/bf16 to fp32."""
    array = np.asarray(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.name in ("bfloat16", "float16"):
        array = array.astype(np.float32)
    return torch.from_numpy(array)


def _to_numpy(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_dtype("bfloat16"))
    return tensor.numpy()


def _cla_gate(global_attn, local_attn, global_logits, local_logits):
    global_scale = torch.sigmoid(global_logits).unsqueeze(-1)
    local_scale = torch.sigmoid(local_logits).unsqueeze(-1)
    return global_attn * global_scale + local_attn * local_scale


def golden_backward(grad_merged, global_attn, local_attn, global_logits, local_logits):
    global_attn = global_attn.detach().requires_grad_(True)
    local_attn = local_attn.detach().requires_grad_(True)
    global_logits = global_logits.detach().requires_grad_(True)
    local_logits = local_logits.detach().requires_grad_(True)
    merged = _cla_gate(global_attn, local_attn, global_logits, local_logits)
    return torch.autograd.grad(
        merged,
        (global_attn, local_attn, global_logits, local_logits),
        grad_merged,
    )


def _kernel_golden(
    grad_merged, global_attn, local_attn, global_logits, local_logits, **kwargs
):
    torch_outputs = golden_backward(
        _to_torch(grad_merged),
        _to_torch(global_attn),
        _to_torch(local_attn),
        _to_torch(global_logits),
        _to_torch(local_logits),
    )
    outputs = [_to_numpy(output) for output in torch_outputs]

    output_dtypes = kwargs.get("output_dtypes") or ()
    if not output_dtypes:
        input_dtype = grad_merged.dtype if hasattr(grad_merged, "dtype") else None
        if input_dtype is not None:
            output_dtypes = (input_dtype,) * len(outputs)
    return [
        output.astype(_numpy_dtype(output_dtypes[index]), copy=False)
        if index < len(output_dtypes)
        else output
        for index, output in enumerate(outputs)
    ]


def cla_gate_backward_golden(
    grad_merged, global_attn, local_attn, global_logits, local_logits, **kwargs
):
    """Legacy kernel entry backed by the autograd golden."""
    return _kernel_golden(
        grad_merged, global_attn, local_attn, global_logits, local_logits, **kwargs
    )
