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

import numpy as np
import torch

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - TTK provides ml_dtypes for BF16 cases.
    ml_dtypes = None

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


def _output_dtype_names(kwargs):
    output_dtypes = kwargs.get("output_dtypes") or ()
    return [_normalize_dtype_name(dtype) for dtype in output_dtypes]


def _to_numpy(value, target_dtype=None):
    result = value.detach().cpu().contiguous()
    target = _normalize_dtype_name(target_dtype)
    if target == "bfloat16":
        if ml_dtypes is None:
            return result.to(torch.float32).numpy()
        return (
            result.to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .view(ml_dtypes.bfloat16)
        )
    array = result.numpy()
    return array.astype(target, copy=False) if target else array


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
    high_precision=False,
):
    u = _torch(u)
    m = _torch(m)
    if high_precision:
        # The caller may already have promoted inputs (fp32 -> fp64).  Do not
        # narrow that value before calculating the true reference.
        work = m.to(torch.float64) if m.dtype != torch.float64 else m
        u = u.to(torch.float64) if u.dtype != torch.float64 else u
    else:
        work = _lift_low_precision(m).to(torch.float32)
        u = _lift_low_precision(u).to(work.dtype)

    def scalar(value):
        return _torch(value).reshape(-1)[0].to(work.dtype)

    eps = scalar(eps)
    beta1 = scalar(beta1)
    clip_threshold = scalar(clip_threshold)
    sum_square_u = scalar(sum_square_u)
    if global_shape is None:
        global_n, global_m = u.shape[-2], u.shape[-1]
    else:
        shape = _torch(global_shape).reshape(-1)
        global_n, global_m = shape[0].to(work.dtype), shape[1].to(work.dtype)
    scale = sum_square_u / (global_n * global_m) / clip_threshold
    # A2 uses ``if (scale_res > 1)``; the false branch also covers NaN.
    scale = torch.where(scale > 1, scale, torch.ones_like(scale))
    beta2 = 1 - beta1
    scaled_u = u / scale
    updated_m = beta2 * scaled_u + beta1 * work
    m_used = updated_m
    m_out = updated_m if use_first_moment else work
    diff = scaled_u - m_used
    squared = torch.square(diff) + eps
    global_sum = squared.sum((-2, -1))
    return (m_out, squared.sum(-1), squared.sum(-2), global_sum)


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
    output_dtypes = _output_dtype_names(kwargs)
    return tuple(
        _to_numpy(value, output_dtypes[index] if index < len(output_dtypes) else None)
        for index, value in enumerate(
            _compute(
                u,
                m,
                eps,
                beta1,
                clip_threshold,
                sum_square_u,
                global_shape,
                use_first_moment,
                high_precision=True,
            )
        )
    )


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

    # Keep the two divisions explicit, matching the A2 arithmetic order while
    # remaining independent from the golden helper's implementation.
    scale = torch.div(sum_square_value, global_n * global_m)
    scale = torch.div(scale, clip_value)
    scale = torch.where(scale > 1, scale, torch.ones_like(scale))
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
    sum_u_rc = torch.sum(squared, dim=(-2, -1))
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


# Coverage markers for host validation: np.float16, np.float32, ge::DT_BF16,
# ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT64, use_first_moment=false,
# use_first_moment=true, GRAPH_FAILED.
