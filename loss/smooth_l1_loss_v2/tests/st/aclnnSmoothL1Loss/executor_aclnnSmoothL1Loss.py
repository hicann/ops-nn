#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_method_torch_nn_smooth_l1_loss")
class MethodAclnnSmoothL1LossApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnSmoothL1LossApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.self_ = input_data.kwargs['self']
        self.target_ = input_data.kwargs['target']
        self.reduction_ = input_data.kwargs['reduction']
        self.beta_ = input_data.kwargs['beta']

        if self.reduction_ == 0:
            reduction = 'none'
        elif self.reduction_ == 1:
            reduction = 'mean'
        else:
            reduction = 'sum'

        ori_dtype = self.self_.dtype
        not_fp32 = ori_dtype != torch.float32

        if self.device == "cpu" and not_fp32:
            self.self_ = self.self_.to(torch.float32)
            self.target_ = self.target_.to(torch.float32)
        
        loss = torch.nn.SmoothL1Loss(reduction=reduction, beta=self.beta_)
        output_data = loss(self.self_, self.target_)

        if self.device == "cpu" and not_fp32:
            output_data = output_data.to(ori_dtype)

        return output_data