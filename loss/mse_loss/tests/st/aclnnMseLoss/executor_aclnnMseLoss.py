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
import torch.nn
import functools
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.common.log import Logger
logging = Logger().get_logger()

@register("ascend_aclnn_mse_loss")
class AclnnMseLossApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self_tensor = input_data.kwargs["self"]
        target_tensor = input_data.kwargs["target"]
        reduction = input_data.kwargs["reduction"]

        reductions = ['none', 'mean', 'sum']
        reduction = reductions[reduction]

        loss = torch.nn.MSELoss(reduction=reduction)

        output = loss(self_tensor.to(torch.float32), target_tensor.to(torch.float32))

        if target_tensor.dtype == self_tensor.dtype:
            output = output.to(self_tensor.dtype)
        return output

