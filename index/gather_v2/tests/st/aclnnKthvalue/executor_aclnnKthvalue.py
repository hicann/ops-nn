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


@register("ascend_aclnnKthvalue")
class aclnnKthvalueExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu':
            x_self = input_data.kwargs["self"]
            x_k = input_data.kwargs["k"]
            x_dim = input_data.kwargs["dim"]
            x_keepdim = input_data.kwargs["keepdim"]
            
            values, indices = torch.kthvalue(x_self, x_k, dim=x_dim, keepdim=x_keepdim)
            return values, indices.zero_()

@register("aclnn_function_Kthvalue")
class aclnnKthvalueExecutor(AclnnBaseApi):
    def after_call(self, output_packages):
        a, b = super().after_call(output_packages)
        return a, b.zero_()
