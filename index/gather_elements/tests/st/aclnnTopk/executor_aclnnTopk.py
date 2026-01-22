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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np


@register("function_aclnnTopk")
class aclnnTopkExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu':
            x_self = input_data.kwargs["self"]
            x_k = input_data.kwargs["k"]
            x_dim = input_data.kwargs["dim"]
            x_largest = input_data.kwargs["largest"]
            x_sorted = input_data.kwargs["sorted"]
            
            values, indices = torch.topk(x_self, k=x_k, dim=x_dim, largest=x_largest, sorted=x_sorted)
            return values, indices.zero_()

@register("aclnn_function_topk")
class aclnnTopkExecutor(AclnnBaseApi):
    def after_call(self, output_packages):
        a, b = super().after_call(output_packages)
        return a, b.zero_()
