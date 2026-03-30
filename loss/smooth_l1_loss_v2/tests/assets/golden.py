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

__golden__ = {"kernel": {"smooth_l1_loss_v2": "smooth_l1_loss_v2_golden"}}


def smooth_l1_loss_v2_golden(predict, label, *, sigma=1.0, reduction='mean', **kwargs):
    '''
    Golden function for smooth_l1_loss_v2.
    All the parameters (names and order) follow @smooth_l1_loss_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    ori_dtype = predict.dtype
    input_dtype = "float32"
    if "float16" in ori_dtype.name:
        predict = predict.astype(input_dtype)
        label = label.astype(input_dtype)
    predict_t = torch.from_numpy(predict)
    label_t = torch.from_numpy(label)
    res = torch.nn.functional.smooth_l1_loss(predict_t, label_t, reduction=reduction, beta=sigma).numpy()
    if "float16" in ori_dtype.name:
        res = res.astype(ori_dtype, copy=False)
    return res
