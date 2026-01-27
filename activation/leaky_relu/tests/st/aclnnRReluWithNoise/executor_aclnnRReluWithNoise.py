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
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset

import os
import random
import numpy as np
import torch
import ctypes

logging = Logger().get_logger()


@register("function_RReluWithNoise")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs['x']
        input_noise = input_data.kwargs['noise']
        input_lower = input_data.kwargs['lower']
        input_upper = input_data.kwargs['upper']
        input_training = input_data.kwargs['training']

        if input_x.dtype == torch.float16:
           input_noise = input_noise.to(torch.float32)
           input_x = input_x.to(torch.float32)
           output_tensor = torch.ops.aten.rrelu_with_noise(input_x, input_noise, lower=input_lower,
                                                               upper=input_upper, training=input_training,
                                                               generator=None).to(torch.float16)
           return output_tensor

        output_tensor = torch.ops.aten.rrelu_with_noise(input_x, input_noise, lower=input_lower, upper=input_upper,
                                                            training=input_training, generator=None)
        return output_tensor