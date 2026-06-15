#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#


import torch
import torch.nn.functional as F
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def hard_sigmoid_reference(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32:
        shifted = torch.clamp(x.to(torch.int64) + 3, min=0, max=6)
        return torch.div(shifted, 6, rounding_mode="trunc").to(torch.int32)
    return F.hardsigmoid(x.float()).to(x.dtype)


@register("aclnn_hardsigmoid_v2")
class TorchHardSigmoidV2(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        return hard_sigmoid_reference(x)
