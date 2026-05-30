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
        "relu6_grad": "relu6_grad_golden"
    }
}


def relu6_grad_golden(gradients, features, **kwargs):
    '''
    Kernel golden for relu6_grad.
    All the parameters follow @relu6_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: dx = (0 < x < 6) ? dy : 0 (element-wise, NumPy broadcasting).
    Strict open-interval: x == 0 and x == 6 both yield 0. NaN in x yields 0
    (any comparison with NaN is false). dy is passed through verbatim when
    the mask is true, so dy = NaN/Inf propagates only inside the (0, 6) band.
    '''
    dtype = gradients.dtype
    if dtype == np.float32:
        x = features
        dy = gradients
    else:
        # half / bf16 path: lift to fp32 for the intermediate compute to
        # match the kernel's Relu6GradFloatCast template; cast back at the end.
        x = features.astype(np.float32)
        dy = gradients.astype(np.float32)
    mask = (x > np.float32(0.0)) & (x < np.float32(6.0))
    out = np.where(mask, dy, np.float32(0.0)).astype(dtype)
    return out
