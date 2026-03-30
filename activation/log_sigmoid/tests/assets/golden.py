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

__golden__ = {"kernel": {"log_sigmoid": "log_sigmoid_golden"}}


def log_sigmoid_golden(input0, **kwargs):
    '''
    Golden function for log_sigmoid.
    All the parameters (names and order) follow @log_sigmoid_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    import functools

    input0_dtype = input0.dtype
    if input0_dtype.name in ["bfloat16", "float16"]:
        input0 = input0.astype(np.float32)
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input0.shape)
    input0 = np.reshape(input0, fuseshape)

    input0_tensor = torch.from_numpy(input0)
    result = torch.nn.LogSigmoid()(input0_tensor)

    result_np = result.detach().numpy().astype(input0_dtype, copy=False)
    return result_np
