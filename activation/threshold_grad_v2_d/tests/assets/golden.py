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
import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor


__golden__ = {
    "aclnn": {
        "aclnnThresholdBackward": "aclnn_threshold_backward_golden",
    },
    "kernel": {"threshold_grad_v2_d": "threshold_grad_v2_d_golden"},
}


def threshold_grad_v2_d_golden(grad_output, self_tensor, *, threshold=1.0, **kwargs):
    """
    Kernel golden for threshold_grad_v2_d.
    All the parameters follow @threshold_grad_v2_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
             input_formats, output_formats, input_ori_formats, output_ori_formats,
             input_dtypes, output_dtypes.
    """
    del kwargs
    grad_output_t = numpy_to_torch_tensor(grad_output)
    self_t = numpy_to_torch_tensor(self_tensor)
    grad_output_t, self_t = torch.broadcast_tensors(grad_output_t, self_t)
    output_dtype = grad_output_t.dtype
    # The kernel promotes every supported dtype to float32 for comparison and
    # selection, then casts the selected gradient back to the output dtype.
    result = torch.ops.aten.threshold_backward(
        grad_output_t.to(torch.float32), self_t.to(torch.float32), threshold
    )
    return torch_to_numpy_tensor(result.to(output_dtype).cpu())


def aclnn_threshold_backward_golden(gradOutput, self, threshold, out, **kwargs):
    if hasattr(threshold, "item"):
        threshold = threshold.item()
    mask = (self > threshold).to(gradOutput.dtype)
    return [gradOutput * mask]


# ----------------------------------------------------------------------------
# E2E 通路（纯新增，上方存量 kernel/aclnn golden 与 __golden__ 注册保持原样，
# 仍由旧机制消费）: torch.ops.aten.threshold_backward 的 NPU 侧由 torch_npu 的
# PrivateUse1 kernel 分派到 aclnnThresholdBackward（本算子交付的 API），golden 侧
# 直接调 ATen CPU kernel，两侧实现来源独立。tolerance 不另行声明，走框架缺省
# （浮点 mix_tolerance / 整型逐位比对）。
# ⚠️ ATen 对整型张量会把 threshold 收窄到张量 dtype（向零截断），而本算子
# int8/uint8 内核按 fp32 与 threshold 比较——负小数阈值时两者语义分叉（如
# self=-2, threshold=-2.5: ATen 判 0，本算子透传 grad），E2E 用例的整型 dtype
# 请用整数值 threshold（浮点 dtype 不受影响）。
# ----------------------------------------------------------------------------
__spec__ = {"torch.ops.aten.threshold_backward": "ThresholdBackwardE2eSpec"}


def _to_float(value):
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def threshold_backward_e2e_golden(gradOutput, selfT, threshold, **kwargs):
    """
    E2E golden for torch.ops.aten.threshold_backward.
    Parameters follow the aten schema (grad_output / self / threshold) without
    outputs; tensors are CPU torch.Tensor bound positionally via the param
    plan. Runs the ATen CPU kernel as the reference against the torch_npu
    dispatch (aten::threshold_backward -> aclnnThresholdBackward).

    Returns:
        [output]
    """
    return [torch.ops.aten.threshold_backward(gradOutput, selfT, _to_float(threshold))]


class ThresholdBackwardE2eSpec:
    golden = staticmethod(threshold_backward_e2e_golden)
