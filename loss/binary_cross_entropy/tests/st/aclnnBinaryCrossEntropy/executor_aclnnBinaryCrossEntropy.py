#!/usr/bin/env python3
# -- coding: utf-8 --
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


@register("ascend_aclnn_binary_cross_entropy")
class MethodBinaryCrossEntropyApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodBinaryCrossEntropyApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):

        ori_dtype = input_data.kwargs['self'].dtype

        x = np.array(input_data.kwargs['self'].to(torch.float32)).astype(np.float32)
        target = np.array(input_data.kwargs['target'].to(torch.float32)).astype(np.float32)
        weight = np.array(input_data.kwargs['weight'].to(torch.float32)).astype(np.float32)
        reduction = np.array(input_data.kwargs['reduction'])

        x_log_tmp = np.log(x)
        x_log_tmp = np.maximum(x_log_tmp, -100)
        data_mul1 = np.multiply(x_log_tmp, target)
        x_neg_tmp = np.multiply(x, -1)
        x1_tmp = np.add(x_neg_tmp, 1)
        y_neg_tmp = np.multiply(target, -1)
        y1_tmp = np.add(y_neg_tmp, 1)
        x1_log_tmp = np.log(x1_tmp)
        x1_log_tmp = np.maximum(x1_log_tmp, -100)
        data_mul2 = np.multiply(x1_log_tmp, y1_tmp)
        data_sum = np.add(data_mul1, data_mul2)
        result = np.multiply(data_sum, -1)        
        result = np.multiply(result, weight)

        reduce_elts = 1.0
        for i in x.shape:
            reduce_elts *= i 
        cof = reduce_elts ** (-1)

        axis_d = []
        for i, _ in enumerate(x.shape):
            axis_d.append(i)
        axis_d = tuple(axis_d)

        if reduction == 1:
            result = np.multiply(result, cof)
            result = np.sum(result, axis = axis_d, keepdims = False)
            result = np.array([result])
        elif reduction == 2:
            result = np.sum(result, axis = axis_d, keepdims = False)
            result = np.array([result])

        return torch.from_numpy(result).to(ori_dtype)