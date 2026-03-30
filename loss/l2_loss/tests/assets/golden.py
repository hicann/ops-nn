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

__golden__ = {"kernel": {"l2_loss": "l2_loss_golden"}}


def l2_loss_golden(x, *, output_dtypes=None, **kwargs):
    '''
    Golden function for l2_loss.
    All the parameters (names and order) follow @l2_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    out_dtype = kwargs.get("output_dtypes", output_dtypes)[0]
    in_shape = x.shape
    in_data = x.astype(out_dtype)
    reduce_axis = tuple([i for i in range(len(in_shape))])
    in_data = in_data*(1/(2**0.5))
    in_data = in_data*in_data
    res = np.sum(in_data, reduce_axis)
    return res
