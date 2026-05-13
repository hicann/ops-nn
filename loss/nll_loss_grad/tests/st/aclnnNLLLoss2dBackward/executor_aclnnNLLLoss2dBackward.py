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
import torch.nn.functional as F
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_aclnn_nllloss2d_backward")
class MethodNllLoss2dBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodNllLoss2dBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):

        ori_dtype = input_data.kwargs['self'].dtype

        gradOutput = input_data.kwargs['gradOutput']
        x = input_data.kwargs['self']
        target = input_data.kwargs['target'].to(torch.int64)
        weight = input_data.kwargs['weight']
        reduction = input_data.kwargs['reduction']
        ignoreIndex = input_data.kwargs['ignoreIndex']
        totalWeight = input_data.kwargs['totalWeight']
        
        ori_dtype = x.dtype
        not_fp32 = ori_dtype != torch.float32

        if self.device == "cpu" and not_fp32:
            gradOutput = gradOutput.to(torch.float32)
            x = x.to(torch.float32)
            weight = weight.to(torch.float32)
            totalWeight = totalWeight.to(torch.float32)
            out = torch.ops.aten.nll_loss2d_backward(gradOutput, x, target, weight, reduction, ignoreIndex, totalWeight)
            out = out.to(ori_dtype)
        else:
            out = torch.ops.aten.nll_loss2d_backward(gradOutput, x, target, weight, reduction, ignoreIndex, totalWeight)

        return out