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
    "kernel": {
        "adaptive_avg_pool2d": "adaptive_avg_pool2d_golden",
        "aclnnAdaptiveAvgPool2d": "adaptive_avg_pool2d_golden",
    }
}


def adaptive_avg_pool2d_golden(x, output_size, *args, **kwargs):
    """
    Kernel golden for adaptive_avg_pool2d.
    All the parameters follow @adaptive_avg_pool2d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn as nn

    def _is_size(v):
        if isinstance(v, (list, tuple)):
            return len(v) in (0, 1, 2)
        if isinstance(v, np.ndarray):
            return v.ndim == 1 and v.size <= 2
        if torch.is_tensor(v):
            return v.ndim == 1 and v.numel() <= 2
        return False

    # extra positional args may carry the aclnn out tensor; re-locate output_size
    if not _is_size(output_size):
        output_size = next((a for a in args if _is_size(a)), None)
    if output_size is None:
        raise ValueError("adaptive_avg_pool2d_golden: output_size not found in args")

    if len(output_size) == 2:
        output_size = tuple(int(v) for v in output_size)
    elif len(output_size) == 1:
        output_size = int(output_size[0])
    elif len(output_size) == 0:
        output_size = (None, None)
    else:
        raise Exception("adaptive_avg_pool2d output_size dim only supports 0,1,2")

    # x may arrive as torch.Tensor (op_api path) or np.ndarray
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(x)
    out_dtype = x.dtype
    # CPU backend has no fp16/bf16 kernel for adaptive_avg_pool2d, compute in fp32
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    m = nn.AdaptiveAvgPool2d(output_size)
    output = m(x)

    output = output.to(out_dtype)
    if is_numpy:
        output = output.numpy()
    return output
