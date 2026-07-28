#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F

__golden__ = {"kernel": {"poisson_nll_loss": "poisson_nll_loss_golden"}}


def poisson_nll_loss_golden(
    input0, input1, *, log_input=True, full=False, eps=1e-8, reduction="mean", **kwargs
):
    """
    Golden for PoissonNllLoss via the competitor interface torch.nn.functional.poisson_nll_loss
    (the reference itself, not a numpy re-implementation of the formula). Attribute names/order
    follow poisson_nll_loss_def.cpp (input, target, log_input, full, eps, reduction).
    Inputs are numpy.ndarray. Compute in fp32 (the kernel casts fp16->fp32 for compute/reduce),
    then cast the result back to the input dtype.
    """
    ori_dtype = input0.dtype
    inp = torch.from_numpy(np.ascontiguousarray(input0)).to(torch.float32)
    tgt = torch.from_numpy(np.ascontiguousarray(input1)).to(torch.float32)

    y = F.poisson_nll_loss(
        inp, tgt, log_input=log_input, full=full, eps=eps, reduction=reduction
    )

    return y.detach().cpu().numpy().astype(ori_dtype, copy=False)
