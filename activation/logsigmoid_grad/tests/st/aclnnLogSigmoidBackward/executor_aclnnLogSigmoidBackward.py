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
import os
import random
import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("aclnn_log_sigmoid_backward")
class FunctionApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "aclnn":
            self_data_dtype = input_data.kwargs["self"].dtype
            input_data.kwargs["buffer"] = torch.tensor([1]).to(self_data_dtype)
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_output = input_data.kwargs['gradOutput']
        input_tensor = input_data.kwargs['self']
        buffer = input_data.kwargs['buffer']

        sigmoid_neg_x = 1 / (1 + torch.exp(input_tensor))  # σ(-x) = 1 / (1 + e^x)
        output_data = grad_output * sigmoid_neg_x           # ∂L/∂x = ∂L/∂y * ∂y/∂x

        return output_data




