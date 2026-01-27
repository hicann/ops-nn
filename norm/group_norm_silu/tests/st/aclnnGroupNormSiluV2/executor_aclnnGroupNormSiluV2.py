#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_group_norm_silu_v2")
class MethodTorchGroupNormSiluV2Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupNormSiluV2Api, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        dtype = x.dtype
        gamma = input_data.kwargs["gamma"]
        beta = input_data.kwargs["beta"]
        group = input_data.kwargs["group"]
        eps = input_data.kwargs["eps"]
        activateSilu = input_data.kwargs["activateSilu"]

        device = "cpu"
        N = x.shape[0]
        C = x.shape[1]
        remaining_dims = x.shape[2:]
        HW = 1
        for size in remaining_dims:
            HW *= size
        output = torch.ops.aten.native_group_norm(x.to(device).to(torch.float64),
                                                gamma.to(device).to(torch.float64), 
                                                beta.to(device).to(torch.float64),
                                                N, C, HW, group, eps)
        out = output[0]
        if activateSilu == True :
            sigmoid_out = 1 / (1 + torch.exp(-out))
            out = out * sigmoid_out
        return out.to(dtype), output[1].to(dtype), output[2].to(dtype)