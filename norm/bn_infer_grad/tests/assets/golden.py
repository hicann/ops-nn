#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np


__golden__ = {"kernel": {"bn_infer_grad": "bn_infer_grad_golden"}}


def bn_infer_grad_golden(grads, scale, batch_variance, *, epsilon=0.0001, **kwargs):
    """
    Compute BNInferGrad through the PyTorch batch-norm inference backward API.

    In inference mode, native_batch_norm_backward calculates:
        grad_input = grad_out * weight / sqrt(running_var + epsilon)

    BNInferGrad does not expose input or running_mean because they do not
    affect grad_input in this mode, so zero placeholders are supplied.
    """
    import torch

    def to_torch(array):
        array = np.asarray(array)
        if array.dtype.name == "bfloat16":
            return torch.from_numpy(array.astype(np.float32).copy()).to(torch.bfloat16)
        return torch.from_numpy(array.copy())

    grads_array = np.asarray(grads)
    grads_torch = to_torch(grads_array)
    scale_torch = to_torch(scale)
    variance_torch = to_torch(batch_variance)
    input_torch = torch.zeros_like(grads_torch)
    running_mean_torch = torch.zeros_like(scale_torch)

    grad_input, _, _ = torch.ops.aten.native_batch_norm_backward(
        grads_torch,
        input_torch,
        scale_torch,
        running_mean_torch,
        variance_torch,
        None,
        None,
        False,
        float(epsilon),
        [True, False, False],
    )
    if grad_input.dtype == torch.bfloat16:
        return (
            grad_input.detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            .astype(grads_array.dtype)
        )
    return grad_input.detach().cpu().numpy()
