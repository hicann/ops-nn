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

def matmul_celoss_backward(input, weight, softmax, target_mask_bool, masked_target, grad):
    # Add the gradient from matching classes.
    arange_1d = torch.arange(start=0, end=softmax.size()[0], device=softmax.device)  # [BT]

    target_mask = target_mask_bool
    softmax_update = 1.0 - target_mask.view(-1).float() # [BT,]
    softmax[arange_1d, masked_target] -= softmax_update  # masked_target_1d的值代表每一行的index
    
    # Finally elementwise multiplication with the output gradients.
    softmax.mul_(grad.unsqueeze(dim=-1)) # [BT, V]

    # 对输入input, weight求导
    target_dtype = weight.dtype
    if target_dtype != softmax.dtype:
        softmax = softmax.to(target_dtype)
    grad_input = torch.matmul(softmax, weight) # [BT, H]
    grad_weight = torch.matmul(softmax.t(), input) # [V, H]
    return grad_input, grad_weight

@register("aclnn_fused_linear_cross_entropy_loss_grad")
class FusedLinearCrossEntropyLossGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        func = matmul_celoss_backward
        
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        # debug
        # print('-----------------------------', flush=True)
        # for k, v in input_data.kwargs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f'{k}: {v.shape} {v.dtype}')
        #     else:
        #         print(f'{k}: {v}')
        # # if input_data.kwargs['input'].dtype != torch.float32:
        # #     torch.save(input_data.kwargs, '/home/jisihuai/workspace/FusedMatmulCelossGrad/cppExtensionInvocation-aclnn/dump.pt')
        # #     print('----------- save success ----------')
        # print(end='', flush=True)
            
        
        output = None
        if with_output:
            output = func(
                input_data.kwargs['input'],
                input_data.kwargs['weight'],
                input_data.kwargs['softmaxOptional'],
                input_data.kwargs['targetMask'],
                input_data.kwargs['maskedTarget'],
                input_data.kwargs['grad']                
            )
        else:
            func(
                input_data.kwargs['input'],
                input_data.kwargs['weight'],
                input_data.kwargs['softmaxOptional'],
                input_data.kwargs['targetMask'],
                input_data.kwargs['maskedTarget'],
                input_data.kwargs['grad']  
            )

        return output
