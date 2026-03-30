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
        "adam_apply_one_with_decay_assign": "adam_apply_one_with_decay_assign_golden"
    }
}


def adam_apply_one_with_decay_assign_golden(input0, input1, input2, input3, input4, 
                                            mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y,
                                            **kwargs):
    '''
    Golden function for adam_apply_one_with_decay_assign.
    All the parameters (names and order) follow @adam_apply_one_with_decay_assign_golden_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    input0_dtype = input0.dtype
    if input0_dtype.name in ("bfloat16", "float16"):
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
        input2 = input2.astype("float32")
        input3 = input3.astype("float32")
        input4 = input4.astype("float32")
        mul0_x = mul0_x.astype("float32")
        mul1_x = mul1_x.astype("float32")
        mul2_x = mul2_x.astype("float32")
        mul3_x = mul3_x.astype("float32")
        mul4_x = mul4_x.astype("float32")
        add2_y = add2_y.astype("float32")
    
    output0 = np.square(input0) * mul3_x + input1 * mul2_x
    output1 = input2 * mul0_x + input0 * mul1_x

    tmp = np.divide(output1, np.sqrt(output0) + add2_y)
    output2 = input3 - (tmp+input3 *mul4_x )* input4

    if input0_dtype.name in ("bfloat16", "float16"):
        return [output0.astype(input0_dtype, copy=False),
                output1.astype(input0_dtype, copy=False),
                output2.astype(input0_dtype, copy=False)]
    else:
        return [output0, output1, output2]
