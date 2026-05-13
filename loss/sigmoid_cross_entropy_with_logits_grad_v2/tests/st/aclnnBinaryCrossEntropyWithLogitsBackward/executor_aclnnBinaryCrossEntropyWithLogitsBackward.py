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

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_aclnn_binary_cross_entropy_with_logits_backward")
class MethodAclnnBinaryCrossEntropyWithLogitsBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnBinaryCrossEntropyWithLogitsBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.grad_output_ = input_data.kwargs['gradOutput']
        self.self_ = input_data.kwargs['self']
        self.self_.requires_grad_(True)
        self.target_ = input_data.kwargs['target']
        self.weight_optional_ = input_data.kwargs['weightOptional']
        self.pos_weight_optional_ = input_data.kwargs['posWeightOptional']
        self.reduction_ = input_data.kwargs['reduction']
        
        re = self.reduction_
        if re == 0:
            reduction = "none"
        elif re == 1:
            reduction = "mean"
        else:
            reduction = "sum"
        
        if reduction == "none":
            loss = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.pos_weight_optional_, weight=self.weight_optional_)
            output_data = loss(self.self_, self.target_)
            output_data.backward(self.grad_output_)
            output_data = self.self_.grad
        else:
            loss = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.pos_weight_optional_, weight=self.weight_optional_)
            output_data = loss(self.self_, self.target_)
            if reduction == "mean":
                grad_output = self.grad_output_.mean()
            else:
                grad_output = self.grad_output_.sum()
            output_data.backward(grad_output)
            output_data = self.self_.grad
        
        return output_data
        
@register("aclnn_binary_cross_entropy_with_logits_backward")
class BinaryCrossEntropyWithLogitsBackwardAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()
    
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []
        
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
        
        dtype = input_data.kwargs['target'].dtype
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(dtype)
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)
        
        return input_args, output_packages