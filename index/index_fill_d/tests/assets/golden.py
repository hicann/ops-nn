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
import torch


__golden__ = {
    "aclnn": {
        "aclnnInplaceIndexFillTensor": "aclnn_inplace_index_fill_tensor_golden",
        "aclnnIndexFillTensor": "aclnn_index_fill_tensor_golden",
    },
    "kernel": {"index_fill_d": "index_fill_d_golden"},
}


def index_fill_d_golden(x, assist1, assist2, **kwargs):
    """
    Golden function for index_fill_d.
    All the parameters (names and order) follow @index_fill_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    output_y = np.where(assist1 > 0, x, assist2)
    return output_y


def aclnn_index_fill_tensor_golden(self, dim, index, value, out=None, **kwargs):
    if hasattr(dim, "item"):
        dim = dim.item()
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index)
    result = self.clone()
    result.index_fill_(dim, index.long(), value)
    return [result]


def aclnn_inplace_index_fill_tensor_golden(selfRef, dim, index, value, **kwargs):
    if hasattr(dim, "item"):
        dim = dim.item()
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index)
    result = selfRef.clone()
    result.index_fill_(dim, index.long(), value)
    return [result]
