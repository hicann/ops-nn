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

__golden__ = {"kernel": {"nll_loss": "nll_loss_golden"}}


def nll_loss_golden(x, target, weight, *, ignore_index=-100, reduction="mean", **kwargs):
    '''
    Golden function for nll_loss.
    All the parameters (names and order) follow @nll_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    if len(x.shape) == 1:
        if target[0] == ignore_index:
            output = 0
        else:
            total_weight_out = np.array([0])
            if reduction == "mean":
                output = -x[target[0]]
            else:
                output = -weight[target[0]] * x[target[0]]
        return output, total_weight_out
    else:
        input_type = x.dtype
        if "bfloat16" in str(x.dtype) or "float16" in str(x.dtype):
            x = x.astype(np.float32)
            target = target.astype(np.int64)
            weight = weight.astype(np.float32)
        
        weight_tensor = torch.from_numpy(weight)
        x_tensor = torch.from_numpy(x)
        target_tensor = torch.from_numpy(target.astype("int64"))
        
        if reduction == "none":
            reduction_int = int(0)
        elif reduction == "sum":
            reduction_int = int(2)
        elif reduction == "mean":
            reduction_int = int(1)
        
        output, total_weight_out = torch.ops.aten.nll_loss_forward(
            x_tensor, target_tensor, weight_tensor, reduction_int, ignore_index
        )
        
        if reduction_int == 0:
            return output.numpy().astype(input_type)
        else:
            total_weight_out = total_weight_out.numpy()
            return output.numpy().astype(input_type), total_weight_out.astype(input_type)
