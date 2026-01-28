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

import random
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_index_put")            
class AclnnIndexPut(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        x_self = input_data.kwargs["input"]
        x_indices = input_data.kwargs["indices"]
        x_values = input_data.kwargs["values"]
        x_accumulate = input_data.kwargs["accumulate"]
        x_self.index_put_(x_indices, x_values, x_accumulate)
        return x_self

@register("index_put_aclnn_api")            
class IndexPutAclnnApi(AclnnBaseApi):  
    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_package in output_packages:
            output.append(self.acl_tensor_to_torch(output_package))
        return output