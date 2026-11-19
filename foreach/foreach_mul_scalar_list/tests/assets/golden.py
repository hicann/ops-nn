#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np


__golden__ = {
    "kernel": {"foreach_mul_scalar_list": "foreach_mul_scalar_list_golden"},
    "aclnn": {"aclnnForeachMulScalarList": "aclnn_foreach_mul_scalar_list_golden"},
}


def foreach_mul_scalar_list_golden(x, scalars, **kwargs):
    """
    Golden function for foreach_mul_scalar_list.
    All the parameters (names and order) follow @foreach_mul_scalar_list_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    """
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)
    scalars_dtype = getattr(scalars, "dtype", None)

    input_pts = []
    scalar_pts = []
    for idx, inp in enumerate(x):
        if "float16" in dtype_str:
            inp = inp.astype(np.float32)
        inp_pt = torch.from_numpy(inp)
        input_pts.append(inp_pt)

        if scalars_dtype and scalars_dtype.name == "int64":
            scalar_tensor = torch.scalar_tensor(scalars[idx], dtype=torch.int64)
        else:
            scalar_tensor = torch.scalar_tensor(scalars[idx], dtype=torch.float32)
        scalar_pts.append(scalar_tensor)

    input_tuple = tuple(input_pt for input_pt in input_pts)
    result_pts = torch._foreach_mul(input_tuple, scalar_pts)

    return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)


def aclnn_foreach_mul_scalar_list_golden(x, scalars, out, **kwargs):
    """ACLNN golden for aclnnForeachMulScalarList (torch/numpy inputs).

    Keep exactly the same semantics as the kernel golden:
    - x dtype containing 'float16' (float16/bfloat16) is computed in float32;
    - scalar is float32 for float x and int64 for int32 x;
    - multiply uses torch._foreach_mul (int32 wrap / float32 rounding);
    - result is cast back to the original x dtype.
    """
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)
    use_torch = isinstance(x[0], torch.Tensor)
    scalar_dtype = torch.int64 if "int" in dtype_str else torch.float32

    input_pts = []
    scalar_pts = []
    for idx, inp in enumerate(x):
        if "float16" in dtype_str:
            inp_pt = (
                inp.to(torch.float32)
                if isinstance(inp, torch.Tensor)
                else torch.from_numpy(inp.astype(np.float32))
            )
        else:
            inp_pt = inp if isinstance(inp, torch.Tensor) else torch.from_numpy(inp)
        input_pts.append(inp_pt)
        scalar_val = (
            int(scalars[idx]) if scalar_dtype == torch.int64 else float(scalars[idx])
        )
        scalar_pts.append(torch.scalar_tensor(scalar_val, dtype=scalar_dtype))

    result_pts = torch._foreach_mul(tuple(input_pts), scalar_pts)
    if use_torch:
        return tuple(result_pt.to(dtype) for result_pt in result_pts)
    return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)
