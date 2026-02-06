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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np

@register("ascend_method_torch_nn_MaxPool3D")
class MethodTorchNnMaxPool3DWithArgmaxApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchNnMaxPool3DWithArgmaxApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu':
            x = input_data.kwargs["self"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]
            m = torch.nn.MaxPool3d(kernel_size=kernelSize, stride=stride, padding=padding, dilation=dilation,
                                   return_indices=True, ceil_mode=ceilMode)
            ori_dtype = x.dtype
            if x.dtype != torch.float:
                x_float = x.to(torch.float)
            else:
                x_float = x
            output, indices = m(x_float)
            if ori_dtype != torch.float:
                output = output.to(ori_dtype)
            indices = indices.to(torch.int32)
        return output, indices