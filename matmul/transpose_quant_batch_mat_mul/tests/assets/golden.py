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
        "e2e": {
            "torch_npu.npu_transpose_quant_batchmatmul": "torch_npu_npu_transpose_quant_batchmatmul_golden"
        }
}

import ml_dtypes
import torch
import numpy as np


def torch_npu_npu_transpose_quant_batchmatmul_golden(x1, x2, dtype, *, bias=None, x1_scale=None,
                                                    x2_scale=None, group_sizes=None, perm_x1=None,
                                                    perm_x2=None, perm_y=None, batch_split_factor=None, **kwargs):
    x1 = torch_to_numpy(x1)
    x2 = torch_to_numpy(x2)
    bias = torch_to_numpy(bias)
    x1_scale = torch_to_numpy(x1_scale)
    x2_scale = torch_to_numpy(x2_scale)
    x1_dtype = str(x1.dtype)
    x2_dtype = str(x2.dtype)
    x1_scale_dtype = str(x1_scale.dtype)
    pertoken_flag = (x1_scale is not None 
                    and x1_dtype in ("int8", "float8_e4m3fn", "float8_e5m2")
                    and x1_scale_dtype in ("float32",))

    x1 = transpose_(x1, perm_x1)
    x2 = transpose_(x2, perm_x2)

    x1 = x_dtype_cope(x1, x1_dtype)
    x2 = x_dtype_cope(x2, x2_dtype)

    if pertoken_flag:
        out = pertoken_calculation(x1, x2, x1_scale, x2_scale, bias, is_bias_epilogue=False, output_dtype=dtype)
    out = transpose_(out, perm_y)
    out = out_dtype_cope(out, dtype)
    return out


def pertoken_calculation(x1, x2, x1_scale, x2_scale, bias, is_bias_epilogue=False, output_dtype=2):
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    out = torch.matmul(x1, x2)
    if bias is not None:
        bias = torch.from_numpy(bias)
    if bias is not None and (bias.dtype == "int32" or (bias.dtype == "float32" and not is_bias_epilogue)):
        out = torch.add(out, bias)
    out = (out * x2_scale.astype(np.float32))

    x1_scale = torch.from_numpy(x1_scale)
    pertoken_scale_slice = torch.unsqueeze(x1_scale, dim=1).to(torch.float32)
    out = out * pertoken_scale_slice
    if bias is not None and is_bias_epilogue:
        out = torch.tensor(out, dtype=torch.float)
        bias = torch.tensor(bias, dtype=torch.float)
        out = torch.add(out, bias)
    return out.numpy()


def x_dtype_cope(x, x_dtype):
    if x_dtype in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.int32)
    return x


def transpose_(x, array_trans_x):
    if isinstance(x, torch.Tensor):
        x = torch.permute(x, array_trans_x)
    elif isinstance(x, np.ndarray):
        x = np.transpose(x, array_trans_x)
    else:
        raise TypeError("This is not a tensor or an array can be transposed")
    return x


def out_dtype_cope(out, output_dtype):
    from ml_dtypes import bfloat16, float8_e4m3fn
    if output_dtype == torch.float32:
        out = out.astype(np.float32)
    elif output_dtype == torch.float16:
        out = out.astype(np.float16)
    elif output_dtype == torch.int8:
        out = np.clip(out, -128, 127)
        out = out.astype(np.int8)
    elif output_dtype == torch.bfloat16:
        out = out.astype(bfloat16)
    elif output_dtype == torch.float8_e4m3fn:
        out = out.astype(float8_e4m3fn)
    elif output_dtype == torch.int32:
        out = out.astype(np.int32)
    else:
        raise TypeError("invalid output dtype")
    return out


def torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float8_e5m2:
            x = x.to(torch.float32).numpy().astype(ml_dtypes.float8_e5m2)
        elif x.dtype == torch.float8_e4m3fn:
            x = x.to(torch.float32).numpy().astype(ml_dtypes.float8_e4m3fn)
        else:
            x = x.numpy()
    return x