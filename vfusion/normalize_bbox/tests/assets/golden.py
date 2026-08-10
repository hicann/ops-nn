#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""NormalizeBBox kernel/GEIR golden (TestSpec multi-path form).

The mathematical golden uses Torch tensor composition and honors TTK Promote:
it computes in the dtype presented by the framework and never narrows a promoted
fp64/fp32 input.  The independent third-party compose mirrors the arch35 kernel's
dtype-dependent int32 -> fp32 [-> fp16] divisor construction and division path.
"""

import numpy as np
import torch


# kernel and GEIR share the operator's snake_case registration key and Spec.
__spec__ = {"normalize_bbox": "NormalizeBBoxKernelSpec"}

# Compatibility entry for the legacy plugin loader.
__golden__ = {
    "kernel": {"normalize_bbox": "normalize_bbox_golden"},
}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """Convert string-valued CSV attributes to the declared default type."""
    value = kwargs.get(name, default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false", "yes", "no", "1", "0"):
            return text in ("true", "yes", "1")
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default
    return value


def _resolve_reversed_box(reversed_box, kwargs):
    """Use the def.cpp snake_case name while preserving legacy aliases."""
    value = reversed_box
    for name in ("reversedBox", "reversed"):
        if kwargs.get(name) is not None:
            value = kwargs[name]
    return _attr({"reversed_box": value}, "reversed_box", False)


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.ascontiguousarray(value))


def _compute(boxes, shape_hw, reversed_box=False):
    """Compute the mathematical reference with Torch competitor composition.

    TTK Promote supplies fp16 inputs as fp32 and fp32 inputs as fp64.  Converting
    ``shape_hw`` to ``boxes.dtype`` therefore raises the integer divisor to the
    active reference precision and never cuts a promoted input back down.
    """
    boxes_tensor = _as_tensor(boxes)
    shape_tensor = _as_tensor(shape_hw)

    batch = boxes_tensor.shape[0]
    hw = shape_tensor.to(dtype=boxes_tensor.dtype)
    height = hw[:, 0]
    width = hw[:, 1]
    divisor4 = torch.stack((height, width, height, width), dim=1)

    rank = boxes_tensor.ndim
    if reversed_box:
        divisor_shape = (batch, 4) + (1,) * (rank - 2)
    else:
        divisor_shape = (batch,) + (1,) * (rank - 2) + (4,)

    # All dimensions are explicit: unlike reshape(batch, -1), this remains valid
    # for batch == 0, which the host tiling accepts as an empty-tensor fast path.
    divisor = torch.reshape(divisor4, divisor_shape)
    return [torch.div(boxes_tensor, divisor)]


def _numpy_outputs(outputs, output_dtypes=None):
    dtypes = output_dtypes or []
    normalized_dtypes = [
        dtype[0] if isinstance(dtype, (list, tuple)) else str(dtype) for dtype in dtypes
    ]
    arrays = []
    for index, output in enumerate(outputs):
        array = output.detach().cpu().numpy()
        if index < len(normalized_dtypes):
            array = array.astype(normalized_dtypes[index], copy=False)
        arrays.append(array)
    return arrays


class _NormalizeBBoxCompose:
    """Independent Torch compose matching the arch35 kernel operation order."""

    def __init__(self, reversed_box=False, **kwargs):
        self.reversed_box = _resolve_reversed_box(reversed_box, kwargs)

    def __call__(self, boxes, shape_hw, **kwargs):
        output_dtype = boxes.dtype
        if output_dtype not in (torch.float16, torch.float32):
            raise TypeError(
                "NormalizeBBox third_party supports only float16/float32 boxes, "
                f"got {output_dtype}"
            )

        # NormalizeBBoxKernel::LoadHW, line for line at dtype level:
        # shape input is int32, then Cast int32 -> fp32, and fp16 boxes take one
        # additional Cast fp32 -> fp16 before Div; fp32 boxes stay in fp32.
        shape_i32 = shape_hw.to(dtype=torch.int32)
        hw_f32 = shape_i32.to(dtype=torch.float32)
        hw_compute = (
            hw_f32.to(dtype=torch.float16) if output_dtype == torch.float16 else hw_f32
        )
        height = hw_compute[:, 0]
        width = hw_compute[:, 1]
        divisor4 = torch.stack((height, width, height, width), dim=1)

        batch = boxes.shape[0]
        rank = boxes.ndim
        if self.reversed_box:
            divisor_shape = (batch, 4) + (1,) * (rank - 2)
        else:
            divisor_shape = (batch,) + (1,) * (rank - 2) + (4,)
        divisor = torch.reshape(divisor4, divisor_shape)

        # NormalizeBBoxKernel::CopyDivide performs Div directly in boxes dtype.
        output = torch.div(boxes, divisor)
        return [output.to(dtype=output_dtype)]


class NormalizeBBoxKernelSpec:
    """Shared kernel + GEIR TestSpec; both pathways receive NumPy inputs."""

    def golden(boxes, shape_hw, reversed_box=False, **kwargs):
        reversed_value = _resolve_reversed_box(reversed_box, kwargs)
        outputs = _compute(boxes, shape_hw, reversed_value)
        return _numpy_outputs(outputs, kwargs.get("output_dtypes"))

    third_party = {"torch": _NormalizeBBoxCompose}
    tolerance = _TOL


def normalize_bbox_golden(boxes, shape_hw, reversed_box=False, **kwargs):
    """Legacy ``__golden__`` entry, backed by the same sole compute core."""
    input_is_tensor = isinstance(boxes, torch.Tensor)
    reversed_value = _resolve_reversed_box(reversed_box, kwargs)
    output = _compute(boxes, shape_hw, reversed_value)[0]
    if input_is_tensor:
        return output
    return output.detach().cpu().numpy()


# 【不存在】aclnn 通路：需求明确不支持，op_host 配置为 ACLNNTYPE aclnn_exclude。
# 【不存在】e2e 通路：需求明确不支持，未提供 torch_npu 到本算子的绑定。
