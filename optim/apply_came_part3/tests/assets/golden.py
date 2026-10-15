#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Torch composition reference for ApplyCamePart3."""

import ml_dtypes
import numpy as np
import torch

__spec__ = {"apply_came_part3": "ApplyCamePart3KernelSpec"}


def _torch(value):
    if isinstance(value, torch.Tensor):
        return value.contiguous()
    array = np.ascontiguousarray(value)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _normalize_dtype_name(dtype):
    if isinstance(dtype, (tuple, list)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }.get(name, name)


def _to_numpy(value):
    result = value.detach().cpu().contiguous()
    if result.dtype == torch.bfloat16:
        bits = result.view(torch.uint16).numpy().copy()
        return bits.view(ml_dtypes.bfloat16)
    return result.numpy()


def _lift_low_precision(value):
    value = _torch(value)
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.to(torch.float32)
    return value


def _compute(
    u,
    m,
    eps,
    beta1,
    clip_threshold,
    sum_square_u,
    global_shape=None,
    use_first_moment=False,
):
    original_m = _torch(m)
    m_dtype = original_m.dtype
    u = _lift_low_precision(u)
    m = _lift_low_precision(original_m)
    compute_dtype = torch.promote_types(u.dtype, m.dtype)
    u = u.to(compute_dtype)
    m = m.to(compute_dtype)
    work = m

    def scalar(value):
        return _torch(value).reshape(-1)[0]

    eps = scalar(eps)
    beta1 = scalar(beta1)
    clip_threshold = scalar(clip_threshold)
    sum_square_u = scalar(sum_square_u)
    if global_shape is None:
        global_n = torch.as_tensor(u.shape[-2], dtype=compute_dtype, device=u.device)
        global_m = torch.as_tensor(u.shape[-1], dtype=compute_dtype, device=u.device)
    else:
        shape = _torch(global_shape).reshape(-1)
        global_n = shape[0].to(compute_dtype)
        global_m = shape[1].to(compute_dtype)
    valid_scale_inputs = (global_n > 0) & (global_m > 0) & (clip_threshold > 0)
    safe_global_n = torch.where(global_n > 0, global_n, torch.ones_like(global_n))
    safe_global_m = torch.where(global_m > 0, global_m, torch.ones_like(global_m))
    safe_clip_threshold = torch.where(
        clip_threshold > 0, clip_threshold, torch.ones_like(clip_threshold)
    )
    raw_scale = sum_square_u / (safe_global_n * safe_global_m) / safe_clip_threshold
    scale = torch.where(
        valid_scale_inputs & (raw_scale > 1), raw_scale, torch.ones_like(raw_scale)
    )
    beta2 = 1 - beta1
    scaled_u = u / scale
    updated_m = beta2 * scaled_u + beta1 * work
    m_used = updated_m
    m_out = updated_m if use_first_moment else work
    diff = scaled_u - m_used
    squared = torch.square(diff) + eps
    global_sum = squared.sum((-2, -1)).reshape(1)
    return (
        m_out.to(m_dtype),
        squared.sum(-1).to(torch.float32),
        squared.sum(-2).to(torch.float32),
        global_sum.to(torch.float32),
    )


def apply_came_part3_golden(
    u,
    m,
    eps,
    beta1,
    clip_threshold,
    sum_square_u,
    global_shape=None,
    use_first_moment=False,
    **kwargs,
):
    return [
        _to_numpy(value)
        for value in (
            _compute(
                u,
                m,
                eps,
                beta1,
                clip_threshold,
                sum_square_u,
                global_shape,
                use_first_moment,
            )
        )
    ]


def _compute_third_party(
    u,
    m,
    eps,
    beta1,
    clip_threshold,
    sum_square_u,
    global_shape=None,
    use_first_moment=False,
):
    """Independent eager Torch composition used by the GPU benchmark.

    This intentionally does not call ``_compute``: the benchmark must remain
    an independent implementation so a shared reference mistake cannot make
    the NPU and GPU results agree spuriously.
    """
    u_tensor = _lift_low_precision(u)
    m_tensor = _lift_low_precision(m)
    compute_dtype = torch.promote_types(u_tensor.dtype, m_tensor.dtype)
    if not compute_dtype.is_floating_point:
        compute_dtype = torch.float32
    u_tensor = u_tensor.to(compute_dtype)
    m_tensor = m_tensor.to(compute_dtype)

    def scalar(value):
        return _lift_low_precision(value).reshape(-1)[0].to(compute_dtype)

    eps_value = scalar(eps)
    beta1_value = scalar(beta1)
    clip_value = scalar(clip_threshold)
    sum_square_value = scalar(sum_square_u)
    if global_shape is None:
        global_n, global_m = u_tensor.shape[-2:]
        global_n = torch.as_tensor(
            global_n, dtype=compute_dtype, device=u_tensor.device
        )
        global_m = torch.as_tensor(
            global_m, dtype=compute_dtype, device=u_tensor.device
        )
    else:
        shape = _torch(global_shape).reshape(-1)
        global_n = shape[0].to(compute_dtype)
        global_m = shape[1].to(compute_dtype)

    valid_scale_inputs = (global_n > 0) & (global_m > 0) & (clip_value > 0)
    safe_global_n = torch.where(global_n > 0, global_n, torch.ones_like(global_n))
    safe_global_m = torch.where(global_m > 0, global_m, torch.ones_like(global_m))
    safe_clip_value = torch.where(
        clip_value > 0, clip_value, torch.ones_like(clip_value)
    )
    raw_scale = torch.div(sum_square_value, safe_global_n * safe_global_m)
    raw_scale = torch.div(raw_scale, safe_clip_value)
    scale = torch.where(
        valid_scale_inputs & (raw_scale > 1), raw_scale, torch.ones_like(raw_scale)
    )
    scaled_u = torch.div(u_tensor, scale)
    updated_m = torch.add(
        torch.mul(scaled_u, 1 - beta1_value),
        torch.mul(m_tensor, beta1_value),
    )
    residual = torch.sub(scaled_u, updated_m)
    squared = torch.add(torch.square(residual), eps_value)
    m_dtype = _torch(m).dtype
    m_output = updated_m if use_first_moment else m_tensor
    sum_u_r = torch.sum(squared, dim=-1)
    sum_u_c = torch.sum(squared, dim=-2)
    sum_u_rc = torch.sum(squared, dim=(-2, -1)).reshape(1)
    return [
        m_output.to(m_dtype),
        sum_u_r.to(torch.float32),
        sum_u_c.to(torch.float32),
        sum_u_rc.to(torch.float32),
    ]


class _ApplyCamePart3Compose:
    _compiled = None
    _compile_failed = False

    def __init__(self, **kwargs):
        cls = type(self)
        if cls._compiled is None and not cls._compile_failed:
            try:
                cls._compiled = torch.compile(
                    _compute_third_party, fullgraph=True, dynamic=False
                )
            except Exception:
                cls._compile_failed = True

    def __call__(
        self,
        u,
        m,
        eps,
        beta1,
        clip_threshold,
        sum_square_u,
        global_shape=None,
        use_first_moment=False,
        **kwargs,
    ):
        args = (
            u,
            m,
            eps,
            beta1,
            clip_threshold,
            sum_square_u,
            global_shape,
            use_first_moment,
        )
        cls = type(self)
        if cls._compiled is not None:
            try:
                return cls._compiled(*args)
            except Exception:
                cls._compiled = None
                cls._compile_failed = True
        return _compute_third_party(*args)


class ApplyCamePart3KernelSpec:
    golden = staticmethod(apply_came_part3_golden)
    third_party = {"torch": _ApplyCamePart3Compose}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }


# Unsupported pathways: ACLNN and Torch E2E are unavailable because this
# operator's CMakeLists.txt declares ACLNNTYPE aclnn_exclude and delivers no
# op_api. TensorFlow, ONNX, and Caffe are also unavailable because the operator
# directory has no framework parser plugin for those frontends.
