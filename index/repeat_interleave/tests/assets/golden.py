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

__golden__ = {"kernel": {"repeat_interleave": "repeat_interleave_golden"}}


def repeat_interleave_golden(x, repeats, *, axis=1000, **kwargs):
    '''
    Golden function for repeat_interleave.
    All the parameters (names and order) follow @repeat_interleave_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    output_shapes = kwargs.get('output_shapes', [[]])
    output_shape = output_shapes[0] if output_shapes else []
    
    repeats_val = repeats.item() if isinstance(repeats, np.ndarray) else repeats
    axis_val = axis % len(x.shape)
    
    if output_shape and repeats_val != output_shape[axis_val]:
        repeats_val = output_shape[axis_val]
    
    dtypes = {'uint64': 'int64', 'uint16': 'int16', 'uint32': 'int32'}
    if x.dtype.name in dtypes.keys():
        input_x = x.view(dtypes[x.dtype.name])
    else:
        input_x = x
    
    x_torch = torch.from_numpy(input_x)
    res_torch = torch.repeat_interleave(x_torch, repeats_val, axis_val)
    return res_torch.numpy().view(x.dtype)
