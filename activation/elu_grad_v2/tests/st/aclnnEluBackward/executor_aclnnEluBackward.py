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

class ExtendedELUBACKWARD(nn.Module):
    def __init__(self, alpha=1.0, scale=1.0, input_scale=1.0, is_result=False):
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        self.input_scale = input_scale
        self.is_result = is_result

    def forward(self, grads, activations):

        if self.is_result:
            mask = activations <= 0
            grads[mask] *= self.input_scale * (activations[mask] + self.alpha * self.scale)
            grads[~mask] *= self.scale
            return grads
        else:
            mask = activations <= 0
            exp_values = torch.exp(activations[mask] * self.input_scale)
            grads[mask] *= self.input_scale * self.alpha * self.scale * exp_values
            grads[~mask] *= self.scale
            return grads


@register("ascend_function_elu_backward")
class MethodEluBackward(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodEluBackward, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_a = input_data.kwargs["gradOutput"]
        
        input_b = input_data.kwargs["alpha"]

        input_c = input_data.kwargs["scale"]

        input_d = input_data.kwargs["inputScale"]

        input_e = input_data.kwargs["isResult"]

        input_f = input_data.kwargs["selfOrResult"]

        ext_elu_backward = ExtendedELUBACKWARD(alpha=input_b, scale=input_c, input_scale=input_d, is_result=input_e)
        output = ext_elu_backward(input_a, input_f)

        return output