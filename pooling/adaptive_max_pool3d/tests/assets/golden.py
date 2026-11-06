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
        "aclnnAdaptiveMaxPool3d": "aclnn_adaptive_max_pool3d_golden",
    },
    "kernel": {"adaptive_max_pool3d": "adaptive_max_pool3d_golden"},
}


def adaptive_max_pool3d_golden(x, output_size, indices_dtype=3, **kwargs):
    """
    Kernel golden for adaptive_max_pool3d.
    All the parameters follow @adaptive_max_pool3d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn as nn

    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)
    out_dtype = x.dtype

    m = nn.AdaptiveMaxPool3d(output_size, True)

    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)  ## torch cpu不支持fp16, bf16

    output, indices = m(x)
    output = output.to(out_dtype)
    indices = indices.to(torch.int32)  ## 与竞品差异
    if is_numpy:
        output = output.numpy()
        indices = indices.numpy()

    return output, indices


def aclnn_adaptive_max_pool3d_golden(
    self, outputSize=0, outputOut=None, indicesOut=None, **kwargs
):
    if hasattr(outputSize, "tolist"):
        outputSize = outputSize.tolist()
    elif isinstance(outputSize, int):
        outputSize = [outputSize] * 3
    if isinstance(self, np.ndarray):
        self = torch.from_numpy(self)
    out_dtype = self.dtype
    x = (
        self.to(torch.float32)
        if self.dtype in (torch.float16, torch.bfloat16)
        else self
    )
    result = torch.nn.functional.adaptive_max_pool3d(x, outputSize, return_indices=True)
    return [
        result[0].to(out_dtype),
        result[1].to(torch.int32),
    ]  ## 与竞品差异: indices为int32
