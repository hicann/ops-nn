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
