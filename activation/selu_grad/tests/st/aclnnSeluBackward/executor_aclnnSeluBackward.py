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
import tensorflow as tf


import ctypes

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

def selu_backward(dy, y):

    res_dtype = y.dtype
    alpha = 1.7580993408473768599402175208123
    scale = 1.0507009873554804934193349852946
    grad_out = dy.clone()
    out = y.clone()

    if res_dtype == torch.float16:
        grad_out.float()
        out.float()
    grad_out = torch.where(y > 0, 
                          grad_out * scale,
                          grad_out * (out + alpha))
    if grad_out.dtype != res_dtype:
        grad_out.to(res_dtype)
    return grad_out
    

@register("function_selu_backward")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input_gradOutput = input_data.kwargs['gradOutput']
        input_outputs = input_data.kwargs['result']

        res = selu_backward(input_gradOutput, input_outputs)
        return res
