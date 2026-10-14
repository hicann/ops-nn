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

import numpy as np


__golden__ = {
    "kernel": {"inplace_add_rms_norm": "inplace_add_rms_norm_golden"},
    "aclnn": {"aclnnInplaceAddRmsNorm": "aclnn_inplace_add_rms_norm_golden"},
    "e2e": {"torch_npu.npu_add_rms_norm_v2": "torch_inplace_add_rms_norm_golden"},
}


def inplace_add_rms_norm_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """Generate y/rstd/x outputs for the Ascend 950 kernel ST."""
    import torch

    input_dtype = x1.dtype
    if input_dtype.name == "bfloat16":
        x1_tensor = torch.from_numpy(x1.view(np.float16)).view(torch.bfloat16)
        x2_tensor = torch.from_numpy(x2.view(np.float16)).view(torch.bfloat16)
        gamma_tensor = torch.from_numpy(gamma.view(np.float16)).view(torch.bfloat16)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        gamma_tensor = torch.from_numpy(gamma)

    y, rstd, x = aclnn_inplace_add_rms_norm_golden(
        x1_tensor, x2_tensor, gamma_tensor, epsilon
    )
    if input_dtype.name == "bfloat16":
        y_output = y.view(torch.float16).numpy().view(input_dtype)
        x_output = x.view(torch.float16).numpy().view(input_dtype)
    else:
        y_output = y.numpy()
        x_output = x.numpy()
    return y_output, rstd.numpy(), x_output


def aclnn_inplace_add_rms_norm_golden(
    x1Ref, x2Ref, gamma, epsilon=1e-6, rstdOut=None, **kwargs
):
    """Ascend 950 normal-mode reference; output order is y/rstd/x."""
    import torch

    # Preserve FP64 inputs supplied by TTK Promote.
    calc_dtype = (
        torch.float64
        if torch.float64 in (x1Ref.dtype, x2Ref.dtype, gamma.dtype)
        else torch.float32
    )
    x = x1Ref.to(calc_dtype) + x2Ref.to(calc_dtype)
    norm_axis_begin = x.dim() - gamma.dim()
    reduce_size = gamma.numel()
    reduce_shape = (*x.shape[:norm_axis_begin], reduce_size)
    rstd_shape = (*x.shape[:norm_axis_begin], *([1] * gamma.dim()))
    rstd = torch.rsqrt(
        x.reshape(reduce_shape).square().mean(-1, keepdim=True) + epsilon
    ).reshape(rstd_shape)
    y = (x * rstd * gamma.to(calc_dtype)).to(x1Ref.dtype)
    return y, rstd, x.to(x1Ref.dtype)


def torch_inplace_add_rms_norm_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """TTK cases must set inplace_input_indexes=(0, 1) to collect y/x."""
    y, rstd, x = aclnn_inplace_add_rms_norm_golden(x1, x2, gamma, epsilon)
    return rstd, y, x
