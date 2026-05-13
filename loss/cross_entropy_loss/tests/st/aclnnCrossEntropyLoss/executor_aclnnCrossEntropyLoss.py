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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import OutputData


def CrossEntropyLossForward(predictions, targets, weight, ignore_index, label_smoothing, reduction):
        input_dtype = torch.float32
        N = predictions.shape[0]
        C = predictions.shape[1]
        if weight is None:
            weight = torch.ones((C,))
        predictions = predictions.to(torch.float32)
        predictions_max = torch.max(predictions, dim=1)[0].unsqueeze(-1)
        log_softmax_probs = predictions - predictions_max - torch.log(torch.sum(torch.exp(predictions - predictions_max), -1, keepdim=True))  # log_softmax结果, [N,C]
        nll_loss = torch.gather(log_softmax_probs, 1, targets.reshape(-1,1)).reshape(-1) # 根据targets按行取出元素, [N]
        weight_yn = torch.gather(weight, 0, targets) # 根据targets取出元素, [N]
        loss_out = -nll_loss * weight_yn # [N]
        # masked_fill操作----方法1
        if ignore_index == None:
            ignore_index = -100
        if ignore_index >= 0: # 传了ignore_index，加判断是因为ignore_index是负数的时候不要误操作了
            ignore_mask = (targets - ignore_index).bool().float()
        else:
            ignore_mask = torch.ones((N,))
        loss_out = loss_out * ignore_mask # [N]
    
        smooth_loss = -torch.sum(log_softmax_probs * weight.unsqueeze(0), -1, keepdim=False) # [N]
        if ignore_index >= 0: # 传了ignore_index，加判断是因为ignore_index是负数的时候不要误操作了
            smooth_loss = smooth_loss * ignore_mask # masked_fill操作
        if reduction == "mean":
            weight_after_mask = weight_yn * ignore_mask # masked_fill操作
            mean_out = torch.sum(loss_out, -1, keepdim=False) / torch.sum(weight_after_mask, -1, keepdim=False)
            ret = (1 - label_smoothing) * mean_out + torch.sum(smooth_loss, -1, keepdim=False) / torch.sum(weight_after_mask, -1, keepdim=False) * label_smoothing / C
        elif reduction == "sum":
            sum_out = torch.sum(loss_out, -1, keepdim=False)
            ret = (1 - label_smoothing) * sum_out + torch.sum(smooth_loss, -1, keepdim=False) * label_smoothing / C
        else:
            none_out = loss_out
            ret = (1 - label_smoothing) * none_out + smooth_loss * label_smoothing / C
        ret = torch.tensor([ret])
        empty_tensor = torch.tensor([1]).to(input_dtype)
        return ret.to(input_dtype), log_softmax_probs.to(input_dtype)

@register("pyaclnn_aclnn_cross_entropy_loss")
class AclnnCrossEntropyLossApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        predictions=input_data.kwargs["input"]
        targets=input_data.kwargs["target"]
        weight=input_data.kwargs["weightOptional"]
        ignore_index=input_data.kwargs["ignoreIndex"]
        label_smoothing=input_data.kwargs["labelSmoothing"]
        reduction=input_data.kwargs["reduction"]
        output = CrossEntropyLossForward(predictions, targets, weight, ignore_index, label_smoothing, reduction)
        return output

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == 'pyaclnn':
            input_dtype = input_data.kwargs["input"].dtype
            self.task_result.output_info_list[0].dtype = str(input_dtype)
            self.task_result.output_info_list[1].dtype = str(input_dtype)
            output_data = {'dtype': str(input_dtype), 'shape': [1]}
            self.task_result.output_info_list.append(OutputData(**output_data))
            self.task_result.output_info_list.append(OutputData(**output_data))


@register("pyaclnn_aclnn_cross_entropy_loss_alcnnbase")
class AclnnCrossEntropyLossApi(AclnnBaseApi):  
    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def init_by_input_data(self, input_data: InputDataset):
        import ctypes
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages
        
    def after_call(self, output_packages):
        output = []
        output_packages = [output_packages[0],output_packages[1]]
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output