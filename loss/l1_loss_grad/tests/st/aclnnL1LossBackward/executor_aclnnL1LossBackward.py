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
from atk.tasks.dataset.base_dataset import OpsDataset

def cal_l1_loss_backward(grad_output, self, target, reduction: int):
    # 计算 sign(self - target)
    diff = self - target
    sign = torch.sign(diff)
    
    # 计算逐元素梯度
    grad_self = grad_output * sign
    
    # 根据 reduction 模式调整梯度
    if reduction == 1:
        grad_self = grad_self * (self.numel() ** -1)    # 平均梯度
    elif reduction == 0 or reduction == 2:
        pass # 保持逐元素梯度
    else:
        raise ValueError("reduction 必须是 0('none')、1('mean') 或 2('sum')")

    return grad_self
    
@register("ascend_aclnn_l1_loss_backward")
class MethodAclnnL1LossBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnL1LossBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.grad_output_ = input_data.kwargs['gradOutput']
        self.self_ = input_data.kwargs['self']
        self.target_ = input_data.kwargs['target']
        self.reduction_ = input_data.kwargs['reduction']

        self.grad_output_, self.self_ = torch.broadcast_tensors(self.grad_output_, self.self_)
        self.grad_output_, self.target_ = torch.broadcast_tensors(self.grad_output_, self.target_)
        
        self.target_.requires_grad_()
        
        output_data = cal_l1_loss_backward(self.grad_output_, self.self_, self.target_, int(self.reduction_))
        return output_data

