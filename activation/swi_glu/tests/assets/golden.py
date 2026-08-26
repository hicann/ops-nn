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
    "kernel": {"swi_glu": "swi_glu_golden"},
    "aclnn": {"aclnnSwiGlu": "aclnn_swi_glu_golden"},
}


def swi_glu_golden(x, *, dim=-1, **kwargs):
    """
    Golden function for swi_glu.
    All the parameters (names and order) follow @swi_glu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch
    import torch.nn.functional as F

    x_dtype = x.dtype
    if "float16" in str(x_dtype):
        x = x.astype(np.float32)

    x_torch = torch.from_numpy(x)
    x_chunks = torch.chunk(x_torch, 2, dim=dim)
    res = F.silu(x_chunks[0]) * x_chunks[1]

    return res.numpy().astype(x_dtype, copy=False)


def aclnn_swi_glu_golden(x, dim, out, **kwargs):
    """
    Aclnn golden for aclnnSwiGlu.
    All the parameters (name & order) follow
        function `aclnnSwiGluGetWorkspaceSize` in aclnn_swi_glu.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch
    import torch.nn.functional as F

    x_dtype = x.dtype
    if "float16" in str(x_dtype):
        x = x.to(torch.float32)

    x_chunks = torch.chunk(x, 2, dim=dim)
    res = F.silu(x_chunks[0]) * x_chunks[1]

    return res.to(x_dtype)
