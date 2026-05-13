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

import random
import torch
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("ascend_fused_linear_online_max_sum")            
class AclnnFusedLinearOnlineMaxSum(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AclnnFusedLinearOnlineMaxSum, self).__init__(task_result)
        self.vocab_parallel_logits_out_flag = None


    def init_by_input_data(self, input_data: InputDataset):
        self.vocab_parallel_logits_out_flag = input_data.kwargs['vocabParallelLogitsOutFlag']


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input = input_data.kwargs['input']
        weight = input_data.kwargs['weight']
        target = input_data.kwargs['target']
        vocab_start_index = input_data.kwargs["vocabStartIndex"]
        vocab_end_index = input_data.kwargs["vocabEndIndex"]

        if self.device == 'cpu':
            vocab_parallel_logits_fp64 = torch.matmul(input.to(torch.float64), weight.t().to(torch.float64))
            vocab_parallel_logits = vocab_parallel_logits_fp64.to(torch.float32)
        else:
            device = "cuda" if self.device == "gpu" else f"{self.device}"
            device = device + f":{self.device_id}"
            vocab_parallel_logits = torch.matmul(input.to(device), weight.to(device).t()).to(torch.float32)

        logits_max_local = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_sub = vocab_parallel_logits - logits_max_local.unsqueeze(dim=-1)

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        pad_num =  (target_mask.shape[0] + 7) // 8 * 8 - target_mask.shape[0]

        if (pad_num > 0):
            target_pad = torch.nn.functional.pad(target, (0, pad_num), "constant", 0)
            target_mask_pad = (target_pad < vocab_start_index) | (target_pad >= vocab_end_index)
        else:
            target_mask_pad = target_mask.clone()
        target_mask_uint8 = target_mask_pad.to(torch.uint8).cpu().numpy().reshape([-1, 8])
        target_mask_uint8_flip = np.flip(target_mask_uint8, axis=1)
        target_mask_out_np = np.packbits(target_mask_uint8_flip, axis=1).reshape(-1)

        partition_vocab_size = vocab_parallel_logits.size()[-1]
        logits_2d = logits_sub.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=input.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        predicted_logits_local = predicted_logits.clone()

        logits_exp = torch.exp(logits_sub)
        sum_exp_logits_local = torch.sum(logits_exp, dim=-1)

        if self.vocab_parallel_logits_out_flag:
            return logits_max_local, sum_exp_logits_local, predicted_logits_local, torch.from_numpy(target_mask_out_np), masked_target, vocab_parallel_logits.to(input.dtype)
        else:
            return logits_max_local, sum_exp_logits_local, predicted_logits_local, torch.from_numpy(target_mask_out_np), masked_target


@register("aclnn_fused_linear_online_max_sum")
class AclnnFusedLinearOnlineMaxSumAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
        vocab_parallel_logits_out_flag = input_args.pop().value

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        
        input_args.extend(output_packages)
        if not vocab_parallel_logits_out_flag:
            input_args.append(TensorPtr())
        
        return input_args, output_packages
