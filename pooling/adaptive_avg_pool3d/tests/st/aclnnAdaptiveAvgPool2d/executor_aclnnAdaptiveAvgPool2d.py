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
# ----------------------------------------------------------------------------
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("ascend_function_adaptive_avg_pool2d")
class AdaptiveAvgPool2dApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AdaptiveAvgPool2dApi, self).__init__(task_result)
        self.input = None
        self.output_size = None
        self.dtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            self.input = self.input.to(torch.float32)
        m = torch.nn.AdaptiveAvgPool2d(self.output_size)
        output = m(self.input)
        if self.device == "cpu":
            output = output.to(self.dtype)
        return output

    def init_by_input_data(self, input_data: InputDataset):
        self.input = input_data.kwargs['input']
        self.dtype = self.input.dtype
        self.output_size = input_data.kwargs['output_size']
    
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if len(input_data.kwargs['input'].shape) == 3:
            return AclFormat.ACL_FORMAT_NCL
        return AclFormat.ACL_FORMAT_NCHW