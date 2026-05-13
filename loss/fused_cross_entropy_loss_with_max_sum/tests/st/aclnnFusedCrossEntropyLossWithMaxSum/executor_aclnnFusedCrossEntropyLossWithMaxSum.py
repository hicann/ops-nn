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
import tensorflow as tf
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import logging


@register("ascend_method_fused_cross_entropy_loss_with_max_sum")
class MethodTFgather(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTFgather, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def matmul_celoss_part2(self, vocab_parallel_logits, logits_max, sum_exp_logits, predicted_logits):
        vocab_parallel_logits = vocab_parallel_logits.to(torch.float32)
        logits_sub = vocab_parallel_logits -logits_max.unsqueeze(dim = -1)
        exp_logits = torch.exp(logits_sub)
        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim = -1))

        return loss, exp_logits

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        vocab_parallel_logits = input_data.kwargs["vocab_parallel_logits"]
        logits_max = input_data.kwargs["logits_max"]
        sum_exp_logits = input_data.kwargs["sum_exp_logits"]
        predicted_logits = input_data.kwargs["predicted_logits"]

        return self.matmul_celoss_part2(vocab_parallel_logits, logits_max, sum_exp_logits, predicted_logits)