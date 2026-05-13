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
import logging
from atk.configs.results_config import TaskResult
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
 
 
 
@register("function_aclnn_cross_entropy_loss_grad_pyaclnn")
class FunctionCrossEntroyLossGradPyaclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        import ctypes
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[4] = ctypes.c_void_p(0)  # 声明一个 void* 类型的变量并初始化为0
        input_args[5] = ctypes.c_void_p(0)  # 声明一个 void* 类型的变量并初始化为0
        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        # 默认调用方式示例
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
 
@register("ascend_function_aclnn_cross_entropy_loss_grad")
class FunctionCrossEntroyLossGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionCrossEntroyLossGradApi, self).__init__(task_result)
 
    def __call__(self, input_data: InputDataset, with_output: bool = False):
 
 
        def crossEntropyLossBackwardNpu(N, C, log_prob, targets, weight, grad, zloss_grad=None, reduction="mean",
                                        ignore_index=-100, label_smoothing=0.0, lse_square_scale=0.0):
            input_dtype = log_prob.dtype
            log_prob = log_prob.to(torch.float32)
            grad = grad.to(torch.float32)
            if weight is None:
                weight = torch.ones(C, )
            weight_yn = torch.gather(weight, 0, targets)  # 根据targets取出元素, [N]
            if ignore_index >= 0:
                ignore_mask = (targets - ignore_index).bool().float()  # [N]
            else:
                ignore_mask = torch.ones((N,))
            if reduction == "mean":
                mean_out_grad = grad * (1 - label_smoothing)  # []
                weight_after_mask = weight_yn * ignore_mask  # [N]
                weight_after_mask_sum = torch.sum(weight_after_mask, -1, keepdim=False)  # []
                smooth_loss_grad = grad / weight_after_mask_sum * label_smoothing / C  # []
                loss_out_grad = mean_out_grad.unsqueeze(-1) / weight_after_mask_sum  # [N]
            elif reduction == "sum":
                sum_out_grad = grad * (1 - label_smoothing)  # []
                smooth_loss_grad = grad.unsqueeze(-1) * label_smoothing / C  # [N]
                loss_out_grad = sum_out_grad.unsqueeze(-1)  # [N]
            else:
                none_out_grad = grad * (1 - label_smoothing)  # [N]
                smooth_loss_grad = grad * label_smoothing / C  # [N]
                loss_out_grad = none_out_grad  # [N]
            loss_out_grad = loss_out_grad * ignore_mask  # [N]
            nll_loss_grad = loss_out_grad * weight_yn  # [N]
            log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_prob) * nll_loss_grad.unsqueeze(-1)
            predictions_grad_loss_out = torch.zeros((N, C)).float()
            for i in range(N):
                predictions_grad_loss_out[i][targets[i]] = nll_loss_grad[i]
            predictions_grad_loss_out = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out
 
            smooth_loss_grad = smooth_loss_grad * ignore_mask  # [N]
            log_softmax_probs_grad_smooth_loss = smooth_loss_grad.unsqueeze(-1) * weight.unsqueeze(0)  # [N,C]
            predictions_grad_smooth_loss = torch.exp(log_prob) * torch.sum(log_softmax_probs_grad_smooth_loss, -1,
                                                                           keepdim=True) - log_softmax_probs_grad_smooth_loss
            predictions_grad = predictions_grad_loss_out + predictions_grad_smooth_loss  # [N,C]+[N,C]->[N,C]
            return predictions_grad.to(input_dtype)
 
 
        grad = input_data.kwargs["gradLoss"]
        log_prob = input_data.kwargs["logProb"]
        targets = input_data.kwargs["targets"]
        weight = input_data.kwargs["weightOptional"]
        reduction = input_data.kwargs["reduction"]
        ignore_index = input_data.kwargs["ignoreIndex"]
        if log_prob.shape == torch.Size([]) or (0 in log_prob.shape): ## 标量tensor和标量tensor场景
            output = log_prob
        else:
            [N, C] = log_prob.shape
            if max([N,C]) >= 2147483649: # 上边界用例
                output = torch.zeros(N, C).to(grad.dtype)
            else:
                output = crossEntropyLossBackwardNpu(N, C, log_prob, targets, weight, grad=grad, zloss_grad=None,
                                             reduction=reduction, ignore_index=ignore_index, label_smoothing=0.0,
                                             lse_square_scale=0.0)
        return output

