#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch

__spec__ = {
    "ascend_requant": "AscendRequantTestSpec",
}


def _ascend_requant_compute(x, req_scale, relu_flag=False):
    scale = req_scale.astype(np.float32)
    scaled = x.astype(np.float32) * scale
    rounded = np.rint(scaled)
    saturated = np.clip(rounded, -128, 127)
    z = saturated.astype(np.int8)
    if relu_flag:
        y = np.maximum(z, 0).astype(np.int8)
    else:
        y = z
    return y


def _ascend_requant_torch_compute(x, req_scale, relu_flag=False):
    scale = req_scale.to(torch.float32)
    scaled = x.to(torch.float32) * scale
    rounded = torch.round(scaled)
    saturated = torch.clamp(rounded, -128, 127)
    z = saturated.to(torch.int8)
    if relu_flag:
        y = torch.clamp(z, min=0).to(torch.int8)
    else:
        y = z
    return y


class AscendRequantTestSpec:
    def golden(x, req_scale, relu_flag=False, **kwargs):
        if "reluFlag" in kwargs:
            relu_flag = kwargs["reluFlag"]
        if isinstance(relu_flag, np.ndarray):
            relu_flag = relu_flag.item()
        relu_flag = bool(relu_flag)

        x = np.asarray(x)
        req_scale = np.asarray(req_scale)
        result = _ascend_requant_compute(x, req_scale, relu_flag)
        return [result]

    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x, req_scale, relu_flag=False, **kwargs):
            if isinstance(relu_flag, np.ndarray):
                relu_flag = relu_flag.item()
            relu_flag = bool(relu_flag)

            x_torch = torch.from_numpy(np.asarray(x))
            req_scale_torch = torch.from_numpy(np.asarray(req_scale))
            result = _ascend_requant_torch_compute(x_torch, req_scale_torch, relu_flag)
            return [result]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "int8": {"standard": "binary_equal"},
    }
