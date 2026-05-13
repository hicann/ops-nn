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


@register("ascend_function_group_norm_swish")
class MethodTorchGroupNormSwishApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupNormSwishApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        gamma = input_data.kwargs["gamma"]
        beta = input_data.kwargs["beta"]
        group = input_data.kwargs["group"]
        eps = input_data.kwargs["eps"]
        activateSwish = input_data.kwargs["activateSwish"]
        swishScale = input_data.kwargs["swishScale"]

        x_dtype = x.dtype
        gamma_dtype = gamma.dtype
    
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
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
        if activateSwish == True :
            sigmoid_out = 1 / (1 + torch.exp(-swishScale * out))
            out = out * sigmoid_out
        return out.to(x_dtype), output[1].to(gamma_dtype), output[2].to(gamma_dtype)
    

    