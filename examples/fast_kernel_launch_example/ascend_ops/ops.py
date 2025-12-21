#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from torch import Tensor
from ascend_ops.trans_conv3d_format import ncdhw_to_ndc1hwc0, ncdhw_to_fz3d, ndc1hwc0_to_ncdhw

__all__ = ["conv3d_custom", ]


def conv3d_custom(input: Tensor, weight: Tensor, strides: list, pads: list, dilations: list,
                  bias: Tensor = None, enable_hf32: bool = False) -> Tensor:
    origin_input_shape = list(input.shape)
    origin_weight_shape = list(weight.shape)
    origin_cout = origin_weight_shape[0]
    input = ncdhw_to_ndc1hwc0(input).npu()
    weight = ncdhw_to_fz3d(weight).npu()
    bias = bias.npu() if bias is not None else None
    output = torch.ops.ascend_ops.conv3d_custom(input, weight, strides, pads, dilations, origin_input_shape,
                                                origin_weight_shape, bias, enable_hf32).cpu()
    return ndc1hwc0_to_ncdhw(output, origin_cout)
