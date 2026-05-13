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
import torch.nn as nn
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import logging

def FatreluMul(input_tensor, threshold):
    
    last_dim = input_tensor.shape[-1]
    if last_dim % 2 == 1:
        return "~~~~~~~~~~shape error~~~~~~~~~~"
    d = last_dim // 2
    
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]

    x1 = torch.threshold(x1, threshold.item(), 0.0)

    return x1 * x2

@register("function_fatrelu_mul")
class MethodTFgather(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTFgather, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        data = input_data.kwargs["input"]
        threshold = input_data.kwargs["threshold"]
        output = FatreluMul(data, threshold)

        return output
