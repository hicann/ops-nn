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
import torch

__golden__ = {
    "aclnn": {
        "aclnnEluBackward": "aclnn_elu_backward_golden",
    },
    "kernel": {"elu_grad_v2": "elu_grad_v2_golden"},
}


def elu_grad_v2_golden(
    grads,
    activations,
    *,
    alpha=1.0,
    scale=1.0,
    input_scale=1.0,
    is_result=False,
    **kwargs,
):
    """
    Golden function for elu_grad_v2.
    All the parameters (names and order) follow @elu_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    grads_dtype = grads.dtype
    if grads_dtype.name in ("bfloat16", "float16"):
        grads = grads.astype(np.float32)
        activations = activations.astype(np.float32)

    grads_torch = torch.from_numpy(grads)
    activations_torch = torch.from_numpy(activations)
    res = torch.ops.aten.elu_backward(
        grads_torch, alpha, scale, input_scale, is_result, activations_torch
    )
    result = res.numpy()

    if grads_dtype.name in ("bfloat16", "float16"):
        result = result.astype(grads_dtype, copy=False)
    return result


def aclnn_elu_backward_golden(
    gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput, **kwargs
):
    alpha_val = alpha.item() if hasattr(alpha, "item") else alpha
    scale_val = scale.item() if hasattr(scale, "item") else scale
    input_scale_val = inputScale.item() if hasattr(inputScale, "item") else inputScale
    is_result = bool(isResult.item()) if hasattr(isResult, "item") else bool(isResult)
    return [
        torch.ops.aten.elu_backward(
            gradOutput, alpha_val, scale_val, input_scale_val, is_result, selfOrResult
        )
    ]
