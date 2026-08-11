#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU golden for ApplyAdamWQuant.

NumPy is used only for TTK array conversion and input sanitization. The golden
math itself is expressed with Torch tensor operations.
"""

import numpy as np
import torch

BLOCK = 256

__spec__ = {
    "apply_adam_w_quant": "ApplyAdamWQuantKernelSpec",
    "aclnnApplyAdamWQuant": "ApplyAdamWQuantAclnnSpec",
}
__golden__ = {
    "kernel": {"apply_adam_w_quant": "apply_adam_w_quant_golden"},
    "aclnn": {"aclnnApplyAdamWQuant": "apply_adam_w_quant_golden"},
}
__input__ = {
    "kernel": {"apply_adam_w_quant": "customize_inputs"},
    "aclnn": {"aclnnApplyAdamWQuant": "customize_inputs"},
}

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "uint8": {"standard": "binary_equal"},
}

_ACLNN_TOLERANCE = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
    "uint8": {"standard": "binary_equal"},
}


def _to_f32(x):
    return torch.as_tensor(np.asarray(x).astype(np.float32), dtype=torch.float32)


def _to_numpy_for_golden(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _quantize_state(state, qmap, absmax):
    one = torch.tensor(1.0, dtype=torch.float32)
    denom = torch.where(absmax == 0, one, absmax)
    normalized = state / denom
    distances = torch.abs(normalized[:, None] - qmap[None, :])
    indices = torch.argmin(distances, dim=1).to(torch.int64)
    if bool(torch.any(qmap < 0)):
        quantized = qmap[indices]
        flip = normalized * quantized < 0
        adjusted = indices.clone()
        adjusted = torch.where(flip & (normalized > 0), adjusted + 1, adjusted)
        adjusted = torch.where(flip & (normalized < 0), adjusted - 1, adjusted)
        indices = torch.clamp(adjusted, 0, qmap.numel() - 1)
    return indices


def apply_adam_w_quant_golden(
    var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, **kwargs
):
    lr = torch.tensor(float(kwargs.get("lr", 0.001)), dtype=torch.float32)
    beta1 = torch.tensor(float(kwargs.get("beta1", 0.9)), dtype=torch.float32)
    beta2 = torch.tensor(float(kwargs.get("beta2", 0.999)), dtype=torch.float32)
    weight_decay = torch.tensor(
        float(kwargs.get("weight_decay", 1.0)), dtype=torch.float32
    )
    eps = torch.tensor(float(kwargs.get("eps", 1e-8)), dtype=torch.float32)
    gnorm_scale = torch.tensor(
        float(kwargs.get("gnorm_scale", 1.0)), dtype=torch.float32
    )
    block_size = int(kwargs.get("block_size", BLOCK))

    out_dtype = np.asarray(var).dtype
    one = torch.tensor(1.0, dtype=torch.float32)
    step_v = _to_f32(step).reshape(-1)[0] + one
    bias_c1 = one - torch.pow(beta1, step_v)
    bias_c2_sqrt = torch.sqrt(one - torch.pow(beta2, step_v))
    step_size = -lr * bias_c2_sqrt / bias_c1
    one_minus_beta1 = one - beta1
    one_minus_beta2 = one - beta2
    weight_decay_factor = one - lr * weight_decay

    var_f = _to_f32(var).reshape(-1).clone()
    grad_f = _to_f32(grad).reshape(-1)
    m_codes = torch.as_tensor(np.asarray(m).reshape(-1).copy(), dtype=torch.int64)
    v_codes = torch.as_tensor(np.asarray(v).reshape(-1).copy(), dtype=torch.int64)
    qmap_m_t = _to_f32(qmap_m).reshape(-1)
    qmap_v_t = _to_f32(qmap_v).reshape(-1)
    absmax_m_t = _to_f32(absmax_m).reshape(-1)
    absmax_v_t = _to_f32(absmax_v).reshape(-1)
    new_am = torch.zeros_like(absmax_m_t, dtype=torch.float32)
    new_av = torch.zeros_like(absmax_v_t, dtype=torch.float32)

    n = var_f.numel()
    num_blocks = (n + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min((block_idx + 1) * block_size, n)
        param = var_f[start:end]
        grad_block = grad_f[start:end] * gnorm_scale
        s1 = qmap_m_t[m_codes[start:end]] * absmax_m_t[block_idx]
        s2 = qmap_v_t[v_codes[start:end]] * absmax_v_t[block_idx]
        s1_update = s1 * beta1 + one_minus_beta1 * grad_block
        s2_update = s2 * beta2 + one_minus_beta2 * (grad_block * grad_block)
        denom = torch.sqrt(s2_update) + eps * bias_c2_sqrt
        param = param + step_size * s1_update / denom
        if float(weight_decay) > 0.0:
            param = param * weight_decay_factor

        abs_m = torch.max(torch.abs(s1_update))
        abs_v = torch.max(s2_update)
        var_f[start:end] = param
        m_codes[start:end] = _quantize_state(s1_update, qmap_m_t, abs_m)
        v_codes[start:end] = _quantize_state(s2_update, qmap_v_t, abs_v)
        new_am[block_idx] = abs_m
        new_av[block_idx] = abs_v

    var_out = var_f.reshape(np.asarray(var).shape).numpy().astype(out_dtype, copy=False)
    return [
        var_out,
        m_codes.numpy()
        .astype(np.asarray(m).dtype, copy=False)
        .reshape(np.asarray(m).shape),
        v_codes.numpy()
        .astype(np.asarray(v).dtype, copy=False)
        .reshape(np.asarray(v).shape),
        new_am.numpy()
        .astype(np.float32, copy=False)
        .reshape(np.asarray(absmax_m).shape),
        new_av.numpy()
        .astype(np.float32, copy=False)
        .reshape(np.asarray(absmax_v).shape),
    ]


def customize_inputs(
    var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step, **kwargs
):
    qmap_m = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    qmap_v = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    m = np.clip(m, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    absmax_m = np.abs(absmax_m).astype(np.float32) + 0.1
    absmax_v = np.abs(absmax_v).astype(np.float32) + 0.1
    step = np.maximum(np.asarray(step), 1)
    return var, grad, m, v, qmap_m, qmap_v, absmax_m, absmax_v, step


class _ApplyAdamWQuantCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *inputs, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        outputs = apply_adam_w_quant_golden(
            *[_to_numpy_for_golden(value) for value in inputs],
            **merged,
        )
        device = inputs[0].device if inputs and torch.is_tensor(inputs[0]) else "cpu"
        return [torch.as_tensor(np.asarray(out), device=device) for out in outputs]


class ApplyAdamWQuantKernelSpec:
    golden = apply_adam_w_quant_golden
    customize_inputs = customize_inputs
    third_party = {"torch": _ApplyAdamWQuantCompose}
    tolerance = _KERNEL_TOLERANCE


class ApplyAdamWQuantAclnnSpec:
    golden = apply_adam_w_quant_golden
    customize_inputs = customize_inputs
    third_party = {"torch": _ApplyAdamWQuantCompose}
    tolerance = _ACLNN_TOLERANCE


ApplyAdamWQuantTestSpec = ApplyAdamWQuantKernelSpec


# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 aclnnApplyAdamWQuant.
