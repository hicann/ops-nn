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
import math
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_aclnn_binary_cross_entropy_backward")
class MethodBinaryCrossEntropyBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodBinaryCrossEntropyBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):
        if input_data.kwargs['self'].dtype == torch.float16:
            input_data.kwargs['gradOutput'] = input_data.kwargs['gradOutput'].to(torch.float32)
            input_data.kwargs['self'] = input_data.kwargs['self'].to(torch.float32)
            input_data.kwargs['target'] = input_data.kwargs['target'].to(torch.float32)
            if len(input_data.kwargs["weightOptional"].shape) != 0:
                input_data.kwargs['weightOptional'] = input_data.kwargs['weightOptional'].to(torch.float32)
            

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        ori_dtype = input_data.kwargs['self'].dtype

        gradOutput = np.array(input_data.kwargs['gradOutput'].to(torch.float32)).astype(np.float32)
        x = np.array(input_data.kwargs['self'].to(torch.float32)).astype(np.float32)
        y = np.array(input_data.kwargs['target'].to(torch.float32)).astype(np.float32)
        reduction = input_data.kwargs['reduction']

        gradOutput = np.broadcast_to(gradOutput, x.shape)
        val1 = np.subtract(x, y)
        minus_predict = np.multiply(x, -1)
        val2_tmp = np.add(minus_predict, 1)
        val2 = np.multiply(x, val2_tmp)
        val2 = np.maximum(val2, 1e-12)
        grad_val = np.multiply(gradOutput, val1)
        res = np.divide(grad_val, val2)
        if len(input_data.kwargs["weightOptional"].shape) != 0:
            weight = np.array(input_data.kwargs['weightOptional'].to(torch.float32)).astype(np.float32)
            res = np.multiply(weight, res)
        
        if reduction == 1:
            reduce_elts = 1.0
            for i in x.shape:
                reduce_elts *= i
            cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
            res = np.multiply(res, cof)
        
        return torch.from_numpy(res).to(ori_dtype)