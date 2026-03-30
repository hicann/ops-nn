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
    "kernel": {
        "foreach_non_finite_check_and_unscale": "foreach_non_finite_check_and_unscale_golden"
    }
}


def foreach_non_finite_check_and_unscale_golden(scaled_grads, found_inf, inv_scale, **kwargs):
    '''
    Golden function for foreach_non_finite_check_and_unscale.
    All the parameters (names and order) follow @foreach_non_finite_check_and_unscale_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    '''
    from ml_dtypes import bfloat16

    inv_scale_val = inv_scale[0] if isinstance(inv_scale, np.ndarray) else inv_scale

    input_dtype = scaled_grads[0].dtype
    input_dtype_str = str(input_dtype)
    output_dtypes = kwargs.get('output_dtypes', [])
    
    # Handle output_dtypes format (may be nested list)
    if output_dtypes:
        if isinstance(output_dtypes[0], (list, tuple)):
            out_dtype_name = output_dtypes[0][0] if output_dtypes[0] else None
        else:
            out_dtype_name = output_dtypes[0]
    else:
        out_dtype_name = None
    
    if out_dtype_name == "bfloat16":
        out_type = bfloat16
    elif out_dtype_name and isinstance(out_dtype_name, str):
        out_type = getattr(np, out_dtype_name)
    else:
        out_type = input_dtype

    found_inf_copy = found_inf.copy()
    has_inf_or_nan = False

    input_pts = []
    for inp in scaled_grads:
        calc_inp = inp.astype(np.float32) if 'bfloat16' in input_dtype_str or 'float16' in input_dtype_str else inp
        
        # Check for inf or nan
        if np.any(np.isinf(calc_inp)) or np.any(np.isnan(calc_inp)):
            has_inf_or_nan = True
        
        inv_scale_res = calc_inp * inv_scale_val
        input_pts.append(inv_scale_res.astype(out_type))

    # Set found_inf flag if any gradient contains inf or nan
    if has_inf_or_nan:
        found_inf_copy[0] = 1.0
    
    input_pts.append(found_inf_copy)
    
    return tuple(input_pts)
