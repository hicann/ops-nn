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
__golden__ = {
        "kernel": {
            "mat_mul_v3": "mat_mul_v3_golden"
        }
}

import torch
import numpy as np
from ml_dtypes import bfloat16

def mat_mul_v3_golden(x1, x2, bias=None, offset_w=None, *, transpose_x1: bool = False,
                      transpose_x2: bool = False, offset_x: int = 0, **kwargs):

    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]

    x1_dtype = x1.dtype.name if hasattr(x1.dtype, 'name') else str(x1.dtype)
    x2_dtype = x2.dtype.name if hasattr(x2.dtype, 'name') else str(x2.dtype)
    bias_dtype = None
    if bias is not None:
        bias_dtype = bias.dtype.name if hasattr(bias.dtype, 'name') else str(bias.dtype)

    x1 = _to_float_tensor(x1, x1_dtype)
    x2 = _to_float_tensor(x2, x2_dtype)

    if transpose_x1:
        array_trans = _gen_axes_for_transpose(len(x1.shape) - 2, [1, 0])
        x1 = x1.permute(*array_trans)
    if transpose_x2:
        array_trans = _gen_axes_for_transpose(len(x2.shape) - 2, [1, 0])
        x2 = x2.permute(*array_trans)

    out = torch.matmul(x1, x2)

    if bias is not None:
        if bias_dtype in ("float16",):
            bias = torch.from_numpy(bias.astype(np.float32))
        elif bias_dtype in ("bfloat16",):
            bias = torch.from_numpy(bias.astype(np.float32))
        else:
            bias = torch.from_numpy(bias.astype(np.float32))
        out = out + bias

    if out_dtype == 'bfloat16':
        out = out.numpy().astype(bfloat16)
    elif out_dtype == 'float16':
        out = out.numpy().astype(np.float16)
    elif out_dtype == 'float32':
        out = out.numpy().astype(np.float32)
    else:
        out = out.numpy().astype(out_dtype)

    return out

def _to_float_tensor(arr, dtype_name):
    if dtype_name in ("float16",):
        arr = torch.from_numpy(arr.astype(np.float32))
    elif dtype_name in ("bfloat16",):
        arr = torch.from_numpy(arr.astype(np.float32))
    elif dtype_name in ("float32",):
        arr = torch.from_numpy(arr)
    else:
        arr = torch.from_numpy(arr.astype(np.float32))
    return arr

def _gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]
