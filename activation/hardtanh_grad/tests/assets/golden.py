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
    "aclnn": {
        "aclnnHardtanhBackward": "aclnn_hardtanh_backward_golden",
    },
    "kernel": {"hardtanh_grad": "hardtanh_grad_golden"},
}


def hardtanh_grad_golden(result, grad, *, min_val=-1.0, max_val=1.0, **kwargs):
    """
    Golden function for hardtanh_grad.
    All the parameters (names and order) follow @hardtanh_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    in_data_type = result.dtype
    if str(in_data_type) == "float16" or str(in_data_type) == "bfloat16":
        grad = grad.astype("float32")
        result = result.astype("float32")

    res = np.where((result <= min_val) | (result >= max_val), 0.0, grad)

    if str(in_data_type) == "float16" or str(in_data_type) == "bfloat16":
        res = res.astype(in_data_type, copy=False)
    return res


def aclnn_hardtanh_backward_golden(gradOutput, self, min, max, out, **kwargs):
    if hasattr(min, "item"):
        min = min.item()
    if hasattr(max, "item"):
        max = max.item()
    mask = (self > min) & (self < max)
    return [gradOutput * mask.to(gradOutput.dtype)]
